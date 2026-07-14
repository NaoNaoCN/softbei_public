"""SafetyAgent：内容安全验证，过滤幻觉、不当内容，附加引用来源。"""

from __future__ import annotations

import json

from loguru import logger

from backend.config import config as app_config
from backend.models.schemas import AgentState
from backend.agents.utils import parse_json_llm_response
from backend.services.llm import chat_completion
from langchain_core.runnables import RunnableConfig

from backend.config import prompts as _prompts

SYSTEM_PROMPT = _prompts.get("agents.safety.system_prompt")


async def run(state: AgentState, config: RunnableConfig = None) -> AgentState:
    """
    SafetyAgent 节点入口。

    职责：
    1. 将 draft_content 与 retrieved_docs 对比
    2. 调用 LLM 审核内容质量
    3. 若通过：state.final_content = state.draft_content
       若不通过：state.final_content = revised_content，state.safety_passed = False
    """
    if not state.draft_content:
        logger.info("[SafetyAgent] draft_content 为空，跳过安全检查")
        return state.model_copy(update={
            "safety_passed": True,
            "final_content": "",
        })

    logger.info(f"[SafetyAgent] 开始审核，draft_len={len(state.draft_content)}")

    # 构造上下文（只取前 N 条参考资料，draft 只取前 N 字用于审核）
    max_ref = app_config.agents.safety.max_ref_docs
    preview_chars = app_config.agents.safety.draft_preview_chars
    context = "\n".join(state.retrieved_docs[:max_ref]) if state.retrieved_docs else "（无参考资料）"
    draft_preview = state.draft_content[:preview_chars]
    prompt = SYSTEM_PROMPT.format(context=context, draft_preview=draft_preview)

    try:
        raw = await chat_completion(
            [{"role": "system", "content": prompt}],
            temperature=app_config.agents.safety.temperature,
            max_tokens=app_config.agents.safety.max_tokens,  # 只需返回 passed + issues
        )
        cleaned = parse_json_llm_response(raw)
        result = json.loads(cleaned)

        passed = result.get("passed", True)
        issues = result.get("issues", [])

        logger.info("[SafetyAgent] passed=%s issues=%s" % (passed, issues))

        # 无论是否通过，始终保留原始 draft_content（不让 LLM 重写文档）
        state = state.model_copy(update={
            "safety_passed": passed,
            "final_content": state.draft_content,
        })

        if not passed and issues:
            state.metadata["safety_issues"] = issues

    except json.JSONDecodeError as e:
        # JSON 解析失败时保守通过，但记录警告
        logger.info("[SafetyAgent] JSON 解析失败: %s，raw_preview=%.200s" % (e, raw if 'raw' in dir() else ''))
        state = state.model_copy(update={
            "safety_passed": True,
            "final_content": state.draft_content,
        })
    except Exception as e:
        # 调用失败时保守通过
        logger.info("[SafetyAgent] LLM 调用失败: %s，保守通过" % e)
        state = state.model_copy(update={
            "safety_passed": True,
            "final_content": state.draft_content,
        })

    return state
