"""QuizAgent：生成多题型测验题目集合。"""

from __future__ import annotations

import json

from loguru import logger

from backend.config import config as app_config
from backend.models.schemas import AgentState, QuestionType
from backend.agents.utils import resolve_kp_name, retrieve_context, safe_json_loads
from backend.services.llm import chat_completion
from langchain_core.runnables import RunnableConfig

from backend.config import prompts as _prompts

SYSTEM_PROMPT = _prompts.get("agents.quiz.system_prompt")


def _get_question_counts(profile) -> tuple[int, int, int]:
    """根据画像薄弱知识点数量决定题目数量分布。"""
    if not profile:
        return tuple(app_config.generation.quiz.counts_default)
    weak_count = len(getattr(profile, "knowledge_weak", []) or [])
    if weak_count > app_config.generation.quiz.weak_threshold_high:
        return tuple(app_config.generation.quiz.counts_high)
    elif weak_count > app_config.generation.quiz.weak_threshold_mid:
        return tuple(app_config.generation.quiz.counts_mid)
    return tuple(app_config.generation.quiz.counts_default)


async def run(state: AgentState, config: RunnableConfig = None) -> AgentState:
    """QuizAgent 节点入口：检索知识点文档并调用 LLM 生成题目 JSON 数组。"""
    kp_name = await resolve_kp_name(state, config)

    # 优先使用 state.question_type_counts（用户指定），否则按 state.num_questions 比例分配
    if state.question_type_counts:
        counts = state.question_type_counts
        single = counts.get("single", 0)
        multi = counts.get("multi", 0)
        fill = counts.get("fill", 0)
        total = single + multi + fill
    elif state.num_questions:
        total = state.num_questions
        single = max(1, total // 2)
        multi = max(1, total // 4)
        fill = max(0, total - single - multi)
    elif state.profile:
        total, single, multi = _get_question_counts(state.profile)
        fill = max(0, total - single - multi)
    else:
        total, single, multi = 4, 2, 1
        fill = max(0, total - single - multi)

    logger.info("[QuizAgent] kp_name=%s total=%d single=%d multi=%d fill=%d"% (
                 kp_name, total, single, multi, fill))

    context, retrieved_texts = await retrieve_context(state, "QuizAgent", config)

    state = state.model_copy(update={"retrieved_docs": retrieved_texts})

    prompt = SYSTEM_PROMPT.format(
        count=total,
        single_count=single,
        multi_count=multi,
        fill_count=fill,
        context=context,
        kp_name=kp_name,
    )

    try:
        raw = await chat_completion(
            [{"role": "system", "content": prompt}],
            temperature=app_config.agents.quiz.temperature,
            max_tokens=app_config.agents.quiz.max_tokens,
        )

        # safe_json_loads 容错：代码块/LaTeX/截断/并排对象
        questions = safe_json_loads(raw)
        if isinstance(questions, dict):
            # LLM 偶尔只返回单个题目对象，包成数组
            questions = [questions]
        logger.info("[QuizAgent] 题目生成成功，共 %d 题" % len(questions))
        draft = json.dumps(questions, ensure_ascii=False)
        state = state.model_copy(update={"draft_content": draft})
    except json.JSONDecodeError as e:
        logger.warning("[QuizAgent] JSON 解析失败: %s，raw_preview=%.200s" % (e, raw if 'raw' in dir() else ''))
        state = state.model_copy(update={"draft_content": "[]"})
    except Exception as e:
        logger.error("[QuizAgent] 生成失败: %s" % e)
        state = state.model_copy(update={"draft_content": f"题目生成失败：{e}"})

    return state
