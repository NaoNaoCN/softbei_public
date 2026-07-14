"""PlannerAgent：解析用户意图，决定生成哪种资源类型并确定目标知识点。"""

from __future__ import annotations

import json

from backend.config import config as app_config
from backend.models.schemas import AgentState, ResourceType
from backend.agents.utils import parse_json_llm_response
from backend.services import profile as profile_svc
from backend.services.llm import chat_completion
from langchain_core.runnables import RunnableConfig
from loguru import logger


from backend.config import prompts as _prompts

SYSTEM_PROMPT = _prompts.get("agents.planner.system_prompt")


_INTENT_CLASSIFY_PROMPT = _prompts.get("agents.planner.intent_classify")


async def run(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    PlannerAgent 节点入口。

    如果 state 中已预设了 resource_type 和 kp_id（直接生成模式），跳过 LLM 分析。
    """
    if state.resource_type and state.kp_id:
        logger.info(
            f"[PlannerAgent] 跳过分析（已预设 resource_type={state.resource_type}, kp_id={state.kp_id}）"
        )
        return state
    db = None
    if config and "configurable" in config:
        db = config["configurable"].get("db")

    profile_ctx = ""
    if state.profile:
        profile_ctx = await profile_svc.build_profile_context(state.profile)
    logger.info(f"[PlannerAgent] profile_ctx={profile_ctx}")

    lookback = app_config.agents.planner.history_lookback_messages
    if state.chat_history:
        # 只有有历史时才需要判断是否为追问
        history_summary = "\n".join(
            f"{m['role']}: {m['content'][:100]}" for m in state.chat_history[-lookback:]
        )
        classify_prompt = _INTENT_CLASSIFY_PROMPT
        classify_messages = [
            {"role": "system", "content": classify_prompt},
        ]
        classify_messages.extend(state.chat_history[-lookback:])
        classify_messages.append({"role": "user", "content": state.user_message})
        try:
            classify_raw = await chat_completion(classify_messages, temperature=app_config.agents.planner.intent_temperature)
            cleaned_classify = parse_json_llm_response(classify_raw)
            classify_result = json.loads(cleaned_classify)
            intent = classify_result.get("intent", "generate")
            if intent == "clarify":
                logger.info(f"[PlannerAgent] 意图分类: clarify，路由到 clarify_agent")
                state = state.model_copy(update={"intent_type": "clarify"})
                return state
        except Exception as e:
            logger.warning(f"[PlannerAgent] 意图分类失败: {e}，默认 generate")

    state = state.model_copy(update={"intent_type": "generate"})

    kp_list = ""
    if db:
        try:
            from sqlalchemy import or_, select as sa_select
            from backend.db.models import KGNode
            # 只查当前用户的知识点 + 公共知识点，上限 500 条防止无界增长
            result = await db.execute(
                sa_select(KGNode)
                .where(
                    or_(
                        KGNode.user_id == state.user_id,
                        KGNode.user_id.is_(None),
                    )
                )
                .limit(500)
            )
            nodes = result.scalars().all()
            kp_list = "\n".join([f"- {n.id}: {n.name}" for n in nodes])
        except Exception:
            kp_list = "（知识点列表获取失败）"

    logger.info(f"[PlannerAgent] Analyzing intent.")
    kp_section = ""
    if kp_list:
        kp_section = f"可用知识点（优先从中选择）：\n{kp_list}"
    prompt = SYSTEM_PROMPT.format(kp_list_section=kp_section)
    messages = [
        {"role": "system", "content": prompt},
    ]
    # 注入对话历史，帮助理解"再来一个"、"换成代码"等指代
    messages.extend(state.chat_history)
    messages.append(
        {"role": "user", "content": f"学生画像：{profile_ctx}\n\n学生需求：{state.user_message}"}
    )

    try:
        raw = await chat_completion(messages, temperature=app_config.agents.planner.classify_temperature)
        # 处理 markdown 代码块包裹的 JSON
        cleaned = parse_json_llm_response(raw)
        result = json.loads(cleaned)
        resource_type_str = result.get("resource_type")
        kp_id = result.get("kp_id")

        if resource_type_str:
            try:
                state = state.model_copy(update={"resource_type": ResourceType(resource_type_str)})
            except ValueError:
                state = state.model_copy(update={"resource_type": None})

        if kp_id:
            state = state.model_copy(update={"kp_id": kp_id})

        # 解析 extra_types（多资源意图）
        extra_types_raw = result.get("extra_types", [])
        if extra_types_raw and isinstance(extra_types_raw, list):
            valid_extra = []
            for et in extra_types_raw:
                try:
                    ResourceType(et)
                    valid_extra.append(et)
                except ValueError:
                    pass
            if valid_extra:
                metadata = dict(state.metadata)
                metadata["extra_resource_types"] = valid_extra
                state = state.model_copy(update={"metadata": metadata})
                logger.info(f"[PlannerAgent] 检测到多资源意图: extra_types={valid_extra}")
    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"[PlannerAgent] LLM 解析失败: {e}, raw={raw if 'raw' in dir() else 'N/A'}")
        # 解析失败时默认生成文档
        state = state.model_copy(update={"resource_type": ResourceType.doc})

    if not state.resource_type:
        state = state.model_copy(update={"resource_type": ResourceType.doc})

    # 确保 kp_id 有值（从用户消息中截取）
    if not state.kp_id:
        state = state.model_copy(update={"kp_id": state.user_message[:app_config.agents.planner.fallback_kp_id_length]})

    logger.info(f"[PlannerAgent] resource_type={state.resource_type}, kp_id={state.kp_id}")

    return state


def route_by_resource_type(state: AgentState) -> str:
    """
    LangGraph 条件路由：根据 intent_type 和 resource_type 决定下一个 Agent 节点名称。
    返回值需与 graph.py 中注册的节点名对应。
    """
    if state.intent_type == "clarify":
        return "clarify_agent"

    mapping = {
        ResourceType.doc: "doc_agent",
        ResourceType.mindmap: "mindmap_agent",
        ResourceType.quiz: "quiz_agent",
        ResourceType.code: "code_agent",
        ResourceType.animation: "anim_agent",
        ResourceType.summary: "summary_agent",
        ResourceType.kg: "kg_agent",
    }
    if state.resource_type and state.resource_type in mapping:
        return mapping[state.resource_type]
    return "recommend_agent"


SMART_PLAN_PROMPT = _prompts.get("agents.planner.smart_plan")


async def plan_resource_types(
    user_id: int,
    kp_id: str,
    db=None,
) -> list[ResourceType]:
    """
    独立调用 planner LLM，根据用户画像和知识点推荐资源类型组合。
    不依赖 LangGraph 图执行。
    """
    profile_ctx = ""
    if db:
        try:
            profile = await profile_svc.get_profile(int(user_id), db)
            if profile:
                profile_ctx = f"专业: {profile.major or '未知'}, 目标: {profile.learning_goal or '未知'}, 认知风格: {profile.cognitive_style or '未知'}, 薄弱点: {profile.knowledge_weak or []}"
        except Exception:
            pass

    kp_name = kp_id
    if db and kp_id.startswith("kp_"):
        try:
            from backend.db.crud import select_one as db_select_one
            from backend.db.models import KGNode
            node = await db_select_one(db, KGNode, filters={"id": kp_id})
            if node:
                kp_name = node.name
        except Exception:
            pass

    messages = [
        {"role": "system", "content": SMART_PLAN_PROMPT},
        {"role": "user", "content": f"学生画像：{profile_ctx}\n目标知识点：{kp_name}"},
    ]

    try:
        raw = await chat_completion(messages, temperature=app_config.agents.planner.smart_plan_temperature)
        cleaned = parse_json_llm_response(raw)
        result = json.loads(cleaned)
        if isinstance(result, list):
            types = []
            for rt_str in result:
                try:
                    types.append(ResourceType(rt_str))
                except ValueError:
                    continue
            if types:
                return types
    except Exception as e:
        logger.warning(f"[plan_resource_types] LLM 解析失败: {e}")

    default_types = []
    for rt_str in app_config.agents.planner.smart_plan_default_types:
        try:
            default_types.append(ResourceType(rt_str))
        except ValueError:
            pass
    return default_types or [ResourceType.doc, ResourceType.quiz]
