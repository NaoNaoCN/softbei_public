"""ProfileAgent：从对话中提取并更新学生画像，判断字段完整性，决定是否放行到 planner。"""

from __future__ import annotations

import json

from langgraph.graph import END
from langchain_core.runnables import RunnableConfig
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import config as app_config, prompts as _prompts
from backend.models.schemas import AgentState, StudentProfileIn, StudentProfileOut
from backend.agents.utils import parse_json_llm_response
from loguru import logger

from backend.services import profile as profile_svc
from backend.services.llm import chat_completion

_EXTRACT_PROMPT = _prompts.get("agents.profile.extract")
_INTENT_PROMPT = _prompts.get("agents.profile.intent")
_ONBOARDING_CLARIFY_PROMPT = _prompts.get("agents.profile.onboarding_clarify")
_RESOURCE_CLARIFY_PROMPT = _prompts.get("agents.profile.resource_clarify")
_NO_DOCS_GUIDE_PROMPT = _prompts.get("agents.profile.no_docs_guide")
_PROFILE_CONFIRM_PROMPT = _prompts.get("agents.profile.profile_confirm")
_PROFILE_CONFIRM_PROMPT_WITH_DOCS = _prompts.get("agents.profile.profile_confirm_with_docs")


def _profile_to_known_fields(profile) -> dict:
    """将 StudentProfileOut 转为非空字段字典。"""
    if profile is None:
        return {}
    data = profile.model_dump(exclude={"id", "user_id", "version", "updated_at"}, exclude_none=True)
    return {k: v for k, v in data.items() if v not in ([], "", None)}


async def _check_user_has_documents(user_id: int) -> bool:
    """检查用户是否在向量库中有已上传的文档。"""
    try:
        from backend.db.vector import get_collection
        col = get_collection()
        results = await col.get(where={"user_id": str(user_id)}, limit=1)
        return bool(results and results.get("ids"))
    except Exception:
        # 向量库不可用时，保守返回 True（不阻断流程）
        return True


def _merge_profile_in_memory(state: AgentState, updates: dict) -> AgentState:
    """将提取的字段合并到 state.profile（内存级别，db 不可用时回退）。"""
    if state.profile is not None:
        existing = state.profile.model_dump()
        for k, v in updates.items():
            if v is not None:
                if isinstance(v, list) and isinstance(existing.get(k), list):
                    existing[k] = list(set(existing[k] + v))
                else:
                    existing[k] = v
        state = state.model_copy(update={"profile": StudentProfileIn(**existing)})
    else:
        # profile 为 None 时，从 updates 创建新画像
        profile_data = {k: v for k, v in updates.items() if v is not None}
        if profile_data:
            state = state.model_copy(update={"profile": StudentProfileIn(**profile_data)})
    return state


def _check_profile_complete(state: AgentState) -> bool:
    """
    判断当前画像是否满足资源生成的最小要求。
    kp_id 由 planner 推断，此处检查 profile 本身是否有足够上下文让 planner 工作。
    最低要求：learning_goal 或 knowledge_weak 至少有一个非空。
    """
    if state.profile is None:
        return False
    p = state.profile
    has_goal = bool(p.learning_goal)
    has_weak = bool(p.knowledge_weak)
    has_mastered = bool(p.knowledge_mastered)
    return has_goal or has_weak or has_mastered


async def run(state: AgentState, config: RunnableConfig) -> AgentState:
    """ProfileAgent 节点入口。"""
    db = None
    if config and "configurable" in config:
        db = config["configurable"].get("db")

    extract_messages = [
        {"role": "system", "content": _EXTRACT_PROMPT},
    ]
    # 注入对话历史，帮助理解上下文
    extract_messages.extend(state.chat_history)
    extract_messages.append({"role": "user", "content": state.user_message})
    try:
        raw = await chat_completion(extract_messages, temperature=app_config.agents.profile.extract_temperature)
        # 处理 markdown 代码块包裹的 JSON
        cleaned = parse_json_llm_response(raw)
        updates = json.loads(cleaned)
        # LLM 可能返回 null / 列表 / 字符串，统一归一为 dict
        if not isinstance(updates, dict):
            updates = {}
    except (json.JSONDecodeError, Exception) as e:
        logger.error(f"画像提取失败: {e}, raw={raw if 'raw' in dir() else 'N/A'}")
        updates = {}

    user_id_int = state.user_id
    logger.info(f"[ProfileAgent] db={db}, config_keys={list(config.keys()) if config else 'None'}")
    if db is not None:
        try:
            state = state.model_copy(update={"profile": await profile_svc.merge_chat_updates(user_id_int, updates, db, user_message=state.user_message)})
        except Exception as e:
            logger.error(f"DB 合并画像失败: {e}")
            # 数据库更新失败时，回退到内存级别合并
            state = _merge_profile_in_memory(state, updates)
    else:
        # 无 db 时使用内存合并
        state = _merge_profile_in_memory(state, updates)

    intent_messages = [
        {"role": "system", "content": _INTENT_PROMPT},
    ]
    # 注入对话历史，帮助理解指代和省略
    intent_messages.extend(state.chat_history)
    intent_messages.append({"role": "user", "content": state.user_message})
    try:
        intent_raw = await chat_completion(intent_messages, temperature=app_config.agents.profile.intent_temperature)
        is_resource_request = intent_raw.strip().lower().startswith("yes")
    except Exception:
        is_resource_request = False

    complete = _check_profile_complete(state)
    state = state.model_copy(update={"profile_complete": complete})

    logger.info(f"[ProfileAgent] updates={updates}")
    logger.info(f"[ProfileAgent] profile={state.profile}")
    logger.info(f"[ProfileAgent] complete={complete}, is_resource_request={is_resource_request}")

    has_user_docs = False
    if complete and is_resource_request:
        has_user_docs = await _check_user_has_documents(state.user_id)
        # 仅在首次对话（无历史）时引导上传，后续用户坚持请求则放行
        is_first_conversation = len(state.chat_history) == 0
        if not has_user_docs and is_first_conversation:
            logger.info(f"[ProfileAgent] 画像完整但用户无已上传文档，引导上传教材")
            known = _profile_to_known_fields(state.profile)
            guide_prompt = _NO_DOCS_GUIDE_PROMPT.format(
                known_fields=json.dumps(known, ensure_ascii=False),
                learning_goal=state.profile.learning_goal or state.user_message[:50],
            )
            try:
                guide_msg = await chat_completion(
                    [{"role": "user", "content": guide_prompt}], temperature=app_config.agents.profile.clarify_temperature
                )
            except Exception:
                guide_msg = (
                    "我已经记录了你的学习画像！不过目前还没有上传课程教材，"
                    "建议你先到「资源库」页面上传相关 PDF 教材，这样我可以基于你的教材生成更精准的学习资源。"
                    "当然，你也可以直接让我生成资源，我会用通用知识来帮你。"
                )
            state = state.model_copy(update={
                "profile_complete": False,  # 阻止路由到 planner
                "clarify_message": guide_msg,
                "final_content": guide_msg,
            })
            return state

    # 有对话历史 = 追问/澄清场景，放行到 planner（由 planner 路由到 clarify_agent）
    # 无对话历史 = 首次自我介绍，确认画像不触发生成
    if complete and not is_resource_request:
        if state.chat_history:
            state = state.model_copy(update={"profile_complete": True})
            return state
        has_user_docs_for_confirm = await _check_user_has_documents(state.user_id)
        known = _profile_to_known_fields(state.profile)
        if not has_user_docs_for_confirm:
            # 无文档：确认画像 + 引导上传
            confirm_prompt = _PROFILE_CONFIRM_PROMPT.format(
                known_fields=json.dumps(known, ensure_ascii=False),
            )
        else:
            # 有文档：确认画像 + 提示可以请求资源
            confirm_prompt = _PROFILE_CONFIRM_PROMPT_WITH_DOCS.format(
                known_fields=json.dumps(known, ensure_ascii=False),
            )
        try:
            confirm_msg = await chat_completion(
                [{"role": "user", "content": confirm_prompt}], temperature=app_config.agents.profile.clarify_temperature
            )
        except Exception:
            if not has_user_docs_for_confirm:
                confirm_msg = (
                    "我已经记录了你的学习画像！建议你先到「资源库」页面上传课程教材 PDF，"
                    "这样我可以基于教材生成更精准的学习资源。"
                    "当然，你也可以直接告诉我想学什么，比如「帮我生成卷积的学习资料」。"
                )
            else:
                confirm_msg = (
                    "我已经记录了你的学习画像！你可以随时告诉我想学什么知识点，"
                    "比如「帮我生成卷积的学习资料」，我会为你生成个性化的学习资源。"
                )
        state = state.model_copy(update={
            "profile_complete": False,  # 阻止路由到 planner
            "clarify_message": confirm_msg,
            "final_content": confirm_msg,
        })
        return state

    if not complete or (is_resource_request and not complete):
        known = _profile_to_known_fields(state.profile)
        missing = []
        if not (state.profile and state.profile.learning_goal):
            missing.append("学习目标")
        if not (state.profile and (state.profile.knowledge_weak or state.profile.knowledge_mastered)):
            missing.append("知识基础（已掌握/薄弱知识点）")
        if not (state.profile and state.profile.cognitive_style):
            missing.append("学习偏好（图文/代码/文字）")

        if is_resource_request:
            topic = state.user_message[:50]
            clarify_prompt = _RESOURCE_CLARIFY_PROMPT.format(
                topic=topic,
                known_fields=json.dumps(known, ensure_ascii=False),
                missing_fields="、".join(missing),
            )
        else:
            clarify_prompt = _ONBOARDING_CLARIFY_PROMPT.format(
                known_fields=json.dumps(known, ensure_ascii=False),
                missing_fields="、".join(missing) if missing else "暂无",
            )

        try:
            clarify_messages = [{"role": "system", "content": clarify_prompt}]
            # 注入历史让追问更自然连贯
            clarify_messages.extend(state.chat_history)
            clarify_messages.append({"role": "user", "content": state.user_message})
            clarify_msg = await chat_completion(clarify_messages, temperature=app_config.agents.profile.clarify_temperature)
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            clarify_msg = "能告诉我你的学习目标和目前的知识基础吗？"

        state = state.model_copy(update={
            "clarify_message": clarify_msg,
            "final_content": clarify_msg,
        })

    return state


def route_after_profile(state: AgentState) -> str:
    """
    profile_agent 出口路由函数。

    - 情况A（纯介绍，无资源请求）或 情况B（有请求但画像不足）→ END
    - 情况C（画像足够）→ "planner_agent"
    """
    if state.profile_complete:
        return "planner_agent"
    return END
