"""
backend/services/llm.py
LLM 调用服务层。
统一封装多个 OpenAI 兼容接口，对 Agent 层屏蔽底层细节。
所有 provider 的 URL、model、超时、重试策略均从 configs/config.yaml 读取。
"""

from __future__ import annotations

from typing import AsyncGenerator, Optional

from loguru import logger

from openai import AsyncOpenAI, RateLimitError, PermissionDeniedError, APIStatusError
from httpx import Timeout, ConnectTimeout, ReadTimeout
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

from backend.config import config

# ===========================================================
# 客户端工厂
# ===========================================================

# 合法 provider 名称，同时也是自动推导 failover 链时的声明顺序（与 config
# providers 块一致）。未在此列表中的名称视为非法，解析链时会 warn 并忽略。
_KNOWN_PROVIDERS = ("spark", "deepseek", "qwen", "openai")


def _get_provider_config(provider: str):
    """根据 provider 名称获取对应的配置。"""
    p = config.llm.providers
    mapping = {
        "spark": p.spark,
        "deepseek": p.deepseek,
        "qwen": p.qwen,
        "openai": p.openai,
    }
    return mapping.get(provider)


def _make_client(provider: str) -> tuple[AsyncOpenAI, str]:
    """
    根据 provider 名称返回 (AsyncOpenAI client, default_model)。
    所有配置从 backend.config 读取。

    Key 解析顺序：provider 专用 key（llm.providers.<name>.api_key）→ 全局 llm.api_key。
    这样单 key 用户无需改动即可运行；多 provider 场景下每家可独立配 key。
    """
    t = config.llm.timeout
    _timeout = Timeout(connect=t.connect, read=t.read, write=t.write, pool=t.pool)

    prov = _get_provider_config(provider)
    if prov and prov.base_url:
        return AsyncOpenAI(
            api_key=prov.api_key or config.llm.api_key,
            base_url=prov.base_url,
            timeout=_timeout,
        ), prov.default_model or config.llm.model

    # fallback: 使用配置中的默认 base_url 和 model
    return AsyncOpenAI(
        api_key=config.llm.api_key,
        base_url=config.llm.base_url,
        timeout=_timeout,
    ), config.llm.model


# ===========================================================
# 核心调用接口
# ===========================================================

def _build_extra_body(provider: str) -> dict | None:
    """构建 extra_body。enable_thinking 是 Qwen3 系列专有参数，
    仅对 qwen provider 传入，避免其他 provider 收到不认识的参数而报 400。"""
    if provider == "qwen" and config.llm.enable_thinking is not None:
        return {"enable_thinking": config.llm.enable_thinking}
    return None


# 需要退避重试的错误特征（不区分大小写匹配异常文本）。
# 讯飞星火把 QPS 限流（11202）、并发限流等包在 HTTP 500 里返回，
# OpenAI SDK 将其解析为 InternalServerError/APIStatusError 而非 RateLimitError，
# 因此按状态码 + 错误文本判定，避免这类"伪 500"限流被当成硬错误直接失败。
_RETRYABLE_ERROR_MARKERS = (
    "qpsoverflow",       # AppIdQpsOverFlowError (11202)
    "11202",
    "concurrency",       # 并发超限
    "flowcontrol",
    "rate limit",
    "too many requests",
)


def _is_retryable_error(exc: BaseException) -> bool:
    """判断异常是否属于可退避重试的类型。

    覆盖两类：
    1. SDK 已归类的限流/超时/连接类异常；
    2. provider 把限流塞进 HTTP 5xx（如讯飞 11202 QpsOverFlow）导致的 APIStatusError。
    """
    if isinstance(exc, (RateLimitError, PermissionDeniedError, TimeoutError,
                        ConnectionError, ConnectTimeout, ReadTimeout)):
        return True
    if isinstance(exc, APIStatusError):
        # 5xx 一律可重试；此外无论状态码，命中限流特征文本也重试。
        if 500 <= getattr(exc, "status_code", 0) < 600:
            return True
        text = str(getattr(exc, "message", "") or exc).lower()
        return any(marker in text for marker in _RETRYABLE_ERROR_MARKERS)
    return False


@retry(
    stop=stop_after_attempt(config.llm.retry.max_attempts),
    wait=wait_exponential(
        multiplier=config.llm.retry.backoff_multiplier,
        min=config.llm.retry.backoff_min_seconds,
        max=config.llm.retry.backoff_max_seconds,
    ),
    retry=retry_if_exception(_is_retryable_error),
)
async def _chat_once_with_retry(
    provider: str,
    messages: list[dict],
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int | None = None,
) -> str:
    """内层：针对【单个 provider】的调用 + 退避重试。

    可重试错误（网络/超时/QPS 11202/5xx）→ 在同一 provider 上退避重试直到
    max_attempts 耗尽；硬错误（鉴权/参数）→ 立即抛出。无论哪种，重试耗尽后
    抛出的异常都由外层 chat_completion 捕获以决定是否切换 provider。
    """
    client, default_model = _make_client(provider)
    _model = model or default_model
    _max_tokens = max_tokens if max_tokens is not None else config.llm.default_max_tokens
    _extra = _build_extra_body(provider)
    response = await client.chat.completions.create(
        model=_model,
        messages=messages,
        temperature=temperature,
        max_tokens=_max_tokens,
        **({} if _extra is None else {"extra_body": _extra}),
    )
    return response.choices[0].message.content or ""


def _has_usable_key(name: str, prov, primary: str) -> bool:
    """判断该 provider 是否有可用 key：主 provider 可回退全局 llm.api_key，
    其余必须配了独立 api_key（否则借用全局 key 调它必然鉴权失败、白费一次切换）。"""
    if name == primary:
        return bool(prov.api_key or config.llm.api_key)
    return bool(prov.api_key)


def _resolve_provider_chain(explicit_provider: Optional[str]) -> list[str]:
    """决定本次调用要依次尝试的 provider 链。

    - 显式传 provider（如 judge.py 交叉验证指定某家）→ 只用该 provider，不 failover。
    - provider=None → 生成 failover 链：
        · llm.provider_order 非空 → 用它作显式覆盖（按其顺序）；
        · 为空 → 按「用户实际配了哪些 key」自动推导：主 provider 打头，
          其余 provider 按声明顺序（_KNOWN_PROVIDERS）追加。
      两种情况都遵守同一过滤规则：只保留合法名称（_KNOWN_PROVIDERS）且有可用 key 的
      provider（见 _has_usable_key）。这样 deepseek/openai 未配 key 时自动排除，
      无需手动维护列表；配了 key 就自动纳入。
    """
    if explicit_provider:
        return [explicit_provider]

    primary = config.llm.provider
    if primary not in _KNOWN_PROVIDERS:
        logger.warning(
            f"[LLM] 主 provider '{primary}' 非法（应为 {_KNOWN_PROVIDERS} 之一），"
            f"failover 可能不可用"
        )

    override = config.llm.provider_order
    if override:
        # 显式覆盖：尊重用户给定顺序。
        candidates = list(override)
    else:
        # 自动推导：主 provider 打头，其余按声明顺序追加。
        candidates = [primary] + [p for p in _KNOWN_PROVIDERS if p != primary]

    chain: list[str] = []
    for name in candidates:
        if name not in _KNOWN_PROVIDERS:
            logger.warning(f"[LLM] provider_order 中的 '{name}' 非法（应为 {_KNOWN_PROVIDERS} 之一），忽略")
            continue
        prov = _get_provider_config(name)
        if prov is None:
            continue
        if not _has_usable_key(name, prov, primary):
            # 显式覆盖里配的项若无 key，提示用户；自动推导时无 key 是正常排除，不刷屏。
            if override:
                logger.warning(f"[LLM] provider '{name}' 未配置可用 api_key，跳过")
            continue
        if name not in chain:
            chain.append(name)
    return chain or [primary]


async def chat_completion(
    messages: list[dict],
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    provider: Optional[str] = None,
) -> str:
    """
    单次非流式对话调用（外层：多 provider 级联 failover）。

    调度：按 llm.provider_order 依次尝试 provider。当前 provider 的退避重试
    全部耗尽（或遇硬错误）后，才切换到下一个——即"重试优先级 > 换 provider"，
    偶发网络抖动由内层退避重试消化，不会因此切换 provider。

    :param messages:    OpenAI 格式消息列表
    :param model:       模型名称，None 则使用各 provider 默认模型
    :param temperature: 温度
    :param max_tokens:  最大输出 token 数，None 则使用配置默认值
    :param provider:    显式指定则只用该 provider（不 failover）；None 则按配置级联
    :return:            模型文本输出
    """
    chain = _resolve_provider_chain(provider)
    last_exc: Exception | None = None
    for idx, _provider in enumerate(chain):
        try:
            return await _chat_once_with_retry(
                _provider, messages, model=model,
                temperature=temperature, max_tokens=max_tokens,
            )
        except Exception as e:  # noqa: BLE001 — 需捕获一切以决定是否切换 provider
            last_exc = e
            nxt = chain[idx + 1] if idx + 1 < len(chain) else None
            if nxt:
                logger.warning(f"[LLM] provider '{_provider}' 失败({e})，切换 '{nxt}'")
            else:
                logger.error(f"[LLM] provider '{_provider}' 失败({e})，已无后备 provider")
    raise last_exc  # type: ignore[misc]


async def stream_chat_completion(
    messages: list[dict],
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    provider: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """流式对话调用，逐 token yield 文本片段。

    注意：本函数【不含】chat_completion 的多 provider 级联 failover——流式一旦
    开始 yield 就无法干净地中途切换 provider。当前全项目无人调用；若future启用，
    需自行在此实现 failover（例如首个 chunk 到达前捕获异常并换 provider 重开流）。
    """
    _provider = provider or config.llm.provider
    client, default_model = _make_client(_provider)
    _model = model or default_model
    _max_tokens = max_tokens if max_tokens is not None else config.llm.default_max_tokens
    _extra = _build_extra_body(_provider)
    stream = await client.chat.completions.create(
        model=_model,
        messages=messages,
        temperature=temperature,
        max_tokens=_max_tokens,
        stream=True,
        **({} if _extra is None else {"extra_body": _extra}),
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content


# Embedding API 客户端（单例复用连接池）
_embedding_client: AsyncOpenAI | None = None


def _get_embedding_client() -> AsyncOpenAI:
    """获取 Embedding API 客户端单例。"""
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = AsyncOpenAI(
            api_key=config.embedding.api_key or config.llm.api_key,
            base_url=config.embedding.api_base_url,
            timeout=Timeout(
                connect=config.embedding.timeout_connect,
                read=config.embedding.timeout_read,
                write=config.embedding.timeout_write,
                pool=config.embedding.timeout_pool,
            ),
        )
    return _embedding_client


async def get_embedding(text: str) -> list[float]:
    """
    获取文本的向量表示。
    调用配置的 Embedding API（模型/地址/key 由 config.embedding 决定，可独立于 LLM provider）。
    失败时返回空向量，由 RAG 层降级为纯 LLM 生成。
    """
    try:
        client = _get_embedding_client()
        response = await client.embeddings.create(
            model=config.embedding.api_model,
            input=text,
        )
        return response.data[0].embedding
    except Exception as e:
        logger.warning(f"[Embedding] API embedding 失败: {e}，返回空向量，RAG 将降级为纯 LLM 生成")
        return []


async def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """
    批量获取文本向量表示。
    单次 API 调用发送多条文本，大幅减少 HTTP 往返次数。
    单次最大条数受 embedding.api_max_batch_size 限制（provider 相关）。
    """
    if not texts:
        return []
    try:
        client = _get_embedding_client()
        response = await client.embeddings.create(
            model=config.embedding.api_model,
            input=texts,
        )
        return [d.embedding for d in response.data]
    except Exception as e:
        logger.error(f"[Embedding] 批量 embedding 失败 ({len(texts)} 条文本): {e}")
        raise


async def check_embedding_health() -> bool:
    """
    检查 Embedding API 连通性。
    发送一条极短文本，使用独立短超时客户端，避免阻塞。
    返回 True 表示 API 可达，False 表示不可达。
    """
    try:
        client = AsyncOpenAI(
            api_key=config.embedding.api_key or config.llm.api_key,
            base_url=config.embedding.api_base_url,
            timeout=Timeout(connect=5, read=5, write=5, pool=5),
        )
        await client.embeddings.create(
            model=config.embedding.api_model,
            input="ping",
        )
        logger.info("[Embedding] 连通性检查通过")
        return True
    except Exception as e:
        logger.warning(f"[Embedding] 连通性检查失败: {e}")
        return False
