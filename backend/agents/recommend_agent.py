"""RecommendAgent：基于学生画像和学习历史推荐下一步学习知识点。"""

from __future__ import annotations

import json

from loguru import logger

from backend.config import config as app_config
from backend.models.schemas import AgentState
from backend.agents.utils import parse_json_llm_response
from backend.services import profile as profile_svc
from backend.services.llm import chat_completion
from backend.db.crud import select as db_select
from langchain_core.runnables import RunnableConfig

from backend.config import prompts as _prompts

SYSTEM_PROMPT = _prompts.get("agents.recommend.system_prompt")


async def run(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    RecommendAgent 节点入口。

    职责：
    1. 从知识图谱查询与已学知识点相邻的节点
    2. 结合画像调用 LLM 选出最优推荐
    3. 将推荐列表存入 state.metadata["recommendations"]
    """
    db = None
    if config and "configurable" in config:
        db = config["configurable"].get("db")

    if state.profile:
        try:
            profile_text = await profile_svc.build_profile_context(state.profile)
        except Exception:
            profile_text = "（暂无画像信息）"
    else:
        profile_text = "（暂无画像信息）"

    mastered = []
    weak = []
    goal = ""
    if state.profile:
        mastered = getattr(state.profile, "knowledge_mastered", []) or []
        weak = getattr(state.profile, "knowledge_weak", []) or []
        goal = getattr(state.profile, "learning_goal", "") or ""

    # 查询可用知识点（按当前用户过滤）
    available_kps = []
    valid_kp_ids: set[str] = set()
    if db:
        try:
            from backend.db.models import KGNode
            from sqlalchemy import select as sa_select, or_
            stmt = sa_select(KGNode).where(
                or_(KGNode.user_id == state.user_id, KGNode.user_id == None)
            )
            result = await db.execute(stmt)
            nodes = result.scalars().all()
            available_kps = [f"- {n.id}: {n.name}" for n in nodes]
            logger.info(f"[RecommendAgent] 从数据库查询到 {len(available_kps)} 个可用知识点")
            valid_kp_ids = {n.id for n in nodes}
        except Exception as exc:
            logger.warning(f"[RecommendAgent] 查询知识图谱失败: {exc}")
            available_kps = ["（知识点列表获取失败）"]
    else:
        available_kps = ["（无数据库连接）"]

    kp_list = "\n".join(available_kps) if available_kps else "（无可用知识点）"
    logger.info("[RecommendAgent] 开始推荐，available_kps=%d goal=%s" % (len(available_kps), goal or "未设定"))

    prompt = SYSTEM_PROMPT.format(
        profile=profile_text,
        mastered=", ".join(mastered) if mastered else "无",
        weak=", ".join(weak) if weak else "无",
        goal=goal or "未设定",
        available_kps=kp_list,
    )

    try:
        raw = await chat_completion(
            [{"role": "user", "content": prompt}],
            temperature=app_config.agents.recommend.temperature,
            max_tokens=app_config.agents.recommend.max_tokens,
        )
        cleaned = parse_json_llm_response(raw)
        recommendations = json.loads(cleaned)
        logger.info(f"[RecommendAgent] 推荐生成成功，共 {len(recommendations) if isinstance(recommendations, list) else 0} 条")

        if not isinstance(recommendations, list):
            recommendations = []

        # 过滤掉 kp_id 不在知识图谱中的虚假推荐
        if valid_kp_ids:
            valid_recs = [r for r in recommendations if r.get("kp_id") in valid_kp_ids]
            if len(valid_recs) < len(recommendations):
                logger.warning(f"[RecommendAgent] 过滤掉 {len(recommendations) - len(valid_recs)} 条无效推荐（kp_id 不存在于知识图谱）")
            recommendations = valid_recs

        new_metadata = dict(state.metadata) if state.metadata else {}
        new_metadata["recommendations"] = recommendations
        new_metadata["kp_name"] = state.kp_id or ""  # 供前端构造路径名

        lines = []
        for i, rec in enumerate(recommendations, 1):
            name = rec.get("kp_name", "未知知识点")
            reason = rec.get("reason", "")
            lines.append(f"**{i}. {name}**")
            if reason:
                lines.append(f"   {reason}\n")
        readable = "\n".join(lines)
        new_metadata["recommendations_text"] = readable

        # 只在没有已生成内容时才写入 final_content
        if state.final_content:
            state = state.model_copy(update={
                "metadata": new_metadata,
                # "final_content": state.final_content + "\n\n---\n\n**推荐下一步学习：**\n" + readable,
                "final_content": state.final_content
            })
        else:
            state = state.model_copy(update={
                "metadata": new_metadata,
                "final_content": "根据你的学习画像，推荐以下学习路径：\n\n" + readable,
            })
    except json.JSONDecodeError as e:
        logger.warning(f"[RecommendAgent] JSON 解析失败: {e}，raw_preview={raw if 'raw' in dir() else ''}")
        new_metadata = dict(state.metadata) if state.metadata else {}
        new_metadata["recommendations"] = []
        state = state.model_copy(update={"metadata": new_metadata})
    except Exception as e:
        logger.error(f"[RecommendAgent] 推荐生成失败: {e}")
        new_metadata = dict(state.metadata) if state.metadata else {}
        new_metadata["recommendations"] = []
        state = state.model_copy(update={"metadata": new_metadata})

    return state
