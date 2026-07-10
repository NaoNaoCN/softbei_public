"""
backend/services/llm.py
LLM 调用服务层。
统一封装多个 OpenAI 兼容接口，对 Agent 层屏蔽底层细节。
所有 provider 的 URL、model、超时、重试策略均从 configs/config.yaml 读取。
"""

from __future__ import annotations

from typing import AsyncGenerator, Optional

from loguru import logger

from openai import AsyncOpenAI, RateLimitError, PermissionDeniedError
from httpx import Timeout, ConnectTimeout, ReadTimeout
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from backend.config import config

# ===========================================================
# 客户端工厂
# ===========================================================

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

@retry(
    stop=stop_after_attempt(config.llm.retry.max_attempts),
    wait=wait_exponential(
        multiplier=config.llm.retry.backoff_multiplier,
        min=config.llm.retry.backoff_min_seconds,
        max=config.llm.retry.backoff_max_seconds,
    ),
    retry=retry_if_exception_type((RateLimitError, PermissionDeniedError, TimeoutError, ConnectionError, ConnectTimeout, ReadTimeout)),
)
async def chat_completion(
    messages: list[dict],
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    provider: Optional[str] = None,
) -> str:
    """
    单次非流式对话调用。

    :param messages:    OpenAI 格式消息列表
    :param model:       模型名称，None 则使用 provider 默认模型
    :param temperature: 温度
    :param max_tokens:  最大输出 token 数，None 则使用配置默认值
    :param provider:    "spark" | "deepseek" | "qwen" | "openai"，None 则读配置文件
    :return:            模型文本输出
    """
    _provider = provider or config.llm.provider
    client, default_model = _make_client(_provider)
    _model = model or default_model
    _max_tokens = max_tokens if max_tokens is not None else config.llm.default_max_tokens
    _extra = _build_extra_body(_provider)
    response = await client.chat.completions.create(
        model=_model,
        messages=messages,
        temperature=temperature,
        max_tokens=_max_tokens,
        **({} if _extra is None else {"extra_body": _extra}),
    )
    return response.choices[0].message.content or ""


async def stream_chat_completion(
    messages: list[dict],
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    provider: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """流式对话调用，逐 token yield 文本片段。"""
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
