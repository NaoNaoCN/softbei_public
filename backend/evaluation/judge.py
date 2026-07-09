"""
backend/evaluation/judge.py
LLM-as-Judge 评估器：用 LLM 评估 RAG 检索和生成质量。

四类评估 Judge：
1. Chunk Relevance  — 检索相关性评分 (0/1/2)
2. Faithfulness     — 生成忠实度（逐句溯源）
3. Completeness     — 知识点完整度（关键概念覆盖）
4. Citation Accuracy— 引用准确性（[n] 标注验证）
"""

from __future__ import annotations

import re
import time
from datetime import datetime
from typing import Optional

from loguru import logger

from backend.config import config
from backend.agents.utils import safe_json_loads
from backend.services.llm import chat_completion
from backend.evaluation.metrics import precision_at_k


# ===========================================================
# Judge Prompt 模板
# ===========================================================

JUDGE_CHUNK_RELEVANCE_PROMPT = """你是一位 RAG 检索质量评估专家。
给定一个用户查询和一条检索到的文本片段，判断该片段是否与查询相关。

查询：{query}
文本片段：{chunk_text}

请判断：
- 2分（高度相关）：该片段直接回答了查询，或包含了查询主题的核心信息
- 1分（部分相关）：该片段与查询话题相关，但不是直接答案
- 0分（无关）：该片段与查询无关

仅返回 JSON（不要包含其他文字）：
{{"score": 0|1|2, "reason": "一句话理由"}}"""


JUDGE_FAITHFULNESS_PROMPT = """你是一位事实核查专家。
给定一段参考资料和 AI 生成的答案，逐句检查答案中的陈述是否能在参考资料中找到依据。

{safety_hints}
参考资料：
{retrieved_docs}

AI 生成答案：
{generated_content}

请将答案拆解为独立的陈述句，逐句判断：
- "supported": 该陈述可以在参考资料中找到直接或间接依据
- "unsupported": 该陈述在参考资料中找不到依据（可能是捏造或来自 LLM 自身知识）

返回 JSON（不要包含其他文字）：
{{
  "statements": [
    {{"text": "陈述原文", "verdict": "supported|unsupported", "evidence": "依据片段或null"}}
  ],
  "faithfulness": 0.0-1.0,
  "issues": ["发现的问题"]
}}"""


JUDGE_COMPLETENESS_ASPECTS_PROMPT = """你是一位课程内容评审专家。
对于知识点"{kp_name}"，请列出该知识点一份合格的学习资料应该包含的关键方面。

要求：
- 列出 4-8 个关键方面
- 每个方面一句话描述
- 覆盖定义、原理、应用、常见误区等维度

返回 JSON（不要包含其他文字）：
{{"aspects": ["方面1", "方面2", ...]}}"""


JUDGE_COMPLETENESS_PROMPT = """你是一位课程内容评审专家。
对于知识点"{kp_name}"，一份合格的学习资料应涵盖以下方面的内容。请评估 AI 生成答案的覆盖程度。

应涵盖的关键方面：
{expected_aspects}

AI 生成答案：
{generated_content}

对每个方面判断是否被覆盖（"covered" / "partial" / "missing"），返回 JSON（不要包含其他文字）：
{{
  "aspects": [
    {{"aspect": "方面名", "coverage": "covered|partial|missing", "evidence": "相关段落或null"}}
  ],
  "completeness": 0.0-1.0
}}"""


JUDGE_CITATION_PROMPT = """你是一位学术引用审核员。
AI 生成的答案中有 [N] 形式的引用标注。请检查引用处的内容是否确实能在对应的参考资料中找到依据。

答案中引用 [{ref_index}] 处的上下文：
{citation_context}

参考资料 [{ref_index}] 的内容：
{reference_chunk}

判断：
- "accurate": 引用处的陈述与参考资料一致，或可在参考资料中找到支撑
- "inaccurate": 引用处的陈述在参考资料中找不到，或含义被明显曲解
- "vague": 引用太模糊或上下文不足，无法做出明确判断

返回 JSON（不要包含其他文字）：
{{"verdict": "accurate|inaccurate|vague", "explanation": "理由"}}"""


# ===========================================================
# RAGJudge 类
# ===========================================================

class RAGJudge:
    """RAG LLM-as-Judge 评估器。

    用法::

        judge = RAGJudge()
        result = await judge.evaluate_full(
            query="什么是梯度下降",
            kp_name="梯度下降",
            retrieved_chunks=chunks,
            generated_content=draft,
        )
    """

    def __init__(
        self,
        provider: str | None = None,
        temperature: float = 0.0,
        sample_rate: float = 0.1,
    ):
        """
        :param provider:    LLM provider，None 使用配置默认值
        :param temperature: 评估用温度（低温度保证一致性）
        :param sample_rate: 采样率（0.0-1.0），用于 decide_sample()
        """
        self.provider = provider or config.llm.provider
        self.temperature = temperature
        self.sample_rate = sample_rate

    # ----------------------------------------------------------
    # Judge 1: Chunk Relevance
    # ----------------------------------------------------------

    async def judge_chunk_relevance(
        self,
        query: str,
        chunk_text: str,
    ) -> dict:
        """
        评估单个 chunk 与查询的相关性。

        :param query:      用户查询
        :param chunk_text: 检索到的文本片段
        :return:           {"score": 0|1|2, "reason": "..."}
        """
        # 截断过长文本（相关性判断不需要全文）
        chunk_preview = chunk_text[:1500]
        prompt = JUDGE_CHUNK_RELEVANCE_PROMPT.format(
            query=query[:500],
            chunk_text=chunk_preview,
        )
        try:
            raw = await chat_completion(
                [{"role": "system", "content": prompt}],
                temperature=self.temperature,
                max_tokens=200,
                provider=self.provider,
            )
            result = safe_json_loads(raw)
            result["score"] = int(result.get("score", 0))
            logger.debug(f"[RAGJudge] chunk_relevance score={result['score']} reason={result.get('reason', '')[:80]}")
            return result
        except Exception as e:
            logger.warning(f"[RAGJudge] chunk_relevance 评估失败: {e}")
            return {"score": 0, "reason": f"评估异常: {e}"}

    async def judge_chunk_relevance_batch(
        self,
        query: str,
        chunks: list,
    ) -> list[int]:
        """
        批量评估多个 chunk 的相关性。

        :param query:  用户查询
        :param chunks: RetrievedChunk 列表（需要有 .text 属性）或纯文本列表
        :return:       相关度标签列表 [0, 2, 1, 0, ...]
        """
        import asyncio

        async def _eval_one(chunk) -> int:
            text = chunk.text if hasattr(chunk, "text") else str(chunk)
            result = await self.judge_chunk_relevance(query, text)
            return result.get("score", 0)

        tasks = [_eval_one(c) for c in chunks]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r if isinstance(r, int) else 0 for r in results]

    # ----------------------------------------------------------
    # Judge 2: Faithfulness
    # ----------------------------------------------------------

    async def judge_faithfulness(
        self,
        retrieved_docs: list[str],
        generated_content: str,
        safety_issues: list[str] | None = None,
    ) -> dict:
        """
        评估生成内容对参考资料的忠实度。

        :param retrieved_docs:    检索到的参考资料文本列表
        :param generated_content: LLM 生成的答案
        :param safety_issues:     SafetyAgent 已发现的问题（可选，用于辅助 Judge 定位疑点）
        :return:                  {"statements": [...], "faithfulness": 0.0-1.0, "issues": [...]}
        """
        if not generated_content:
            return {"statements": [], "faithfulness": 1.0, "issues": []}
        if not retrieved_docs:
            return {
                "statements": [],
                "faithfulness": 0.0,
                "issues": ["无参考资料可供比对，无法评估忠实度"],
            }

        # 截断参考资料和生成内容以控制 token
        docs_text = "\n\n---\n\n".join(
            doc[:1200] for doc in retrieved_docs[:5]
        )
        content_preview = generated_content[:3000]

        # 构造 SafetyAgent 提示（如果有已发现的问题，作为 Judge 的辅助线索）
        safety_hints = ""
        if safety_issues:
            issues_text = "\n".join(f"- {issue}" for issue in safety_issues)
            safety_hints = (
                "【SafetyAgent 已发现的可疑问题（请重点核查以下陈述）】\n"
                f"{issues_text}\n\n"
            )

        prompt = JUDGE_FAITHFULNESS_PROMPT.format(
            safety_hints=safety_hints,
            retrieved_docs=docs_text,
            generated_content=content_preview,
        )
        try:
            raw = await chat_completion(
                [{"role": "system", "content": prompt}],
                temperature=self.temperature,
                max_tokens=2500,
                provider=self.provider,
            )
            result = safe_json_loads(raw)
            # safe_json_loads 可能返回 dict 或 list，此处需要 dict
            if isinstance(result, list):
                logger.warning(f"[RAGJudge] faithfulness 返回了数组而非对象，取首个元素")
                result = result[0] if result else {}
            if not isinstance(result, dict):
                raise ValueError(f"faithfulness judge 返回了非预期的类型: {type(result).__name__}")
            n_stmts = len(result.get("statements", []))
            n_unsupported = sum(1 for s in result.get("statements", []) if s.get("verdict") == "unsupported")
            logger.info(
                f"[RAGJudge] faithfulness={result.get('faithfulness', 0):.2f} "
                f"statements={n_stmts} unsupported={n_unsupported}"
            )
            return result
        except Exception as e:
            logger.warning(f"[RAGJudge] faithfulness 评估失败: {e}")
            return {"statements": [], "faithfulness": 0.0, "issues": [f"评估异常: {e}"]}

    # ----------------------------------------------------------
    # Judge 3: Completeness
    # ----------------------------------------------------------

    async def _generate_expected_aspects(self, kp_name: str) -> list[str]:
        """用 LLM 动态生成知识点应包含的关键方面。"""
        prompt = JUDGE_COMPLETENESS_ASPECTS_PROMPT.format(kp_name=kp_name)
        try:
            raw = await chat_completion(
                [{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=500,
                provider=self.provider,
            )
            result = safe_json_loads(raw)
            return result.get("aspects", [])
        except Exception as e:
            logger.warning(f"[RAGJudge] 生成 expected_aspects 失败: {e}")
            return []

    async def judge_completeness(
        self,
        kp_name: str,
        generated_content: str,
        expected_aspects: list[str] | None = None,
    ) -> dict:
        """
        评估生成内容对知识点关键方面的覆盖完整度。

        :param kp_name:           知识点名称
        :param generated_content: LLM 生成的答案
        :param expected_aspects:  预期应涵盖的方面列表，None 则自动生成
        :return:                  {"aspects": [...], "completeness": 0.0-1.0}
        """
        if not generated_content:
            return {"aspects": [], "completeness": 0.0}

        # 若未提供 expected_aspects，自动生成
        if expected_aspects is None:
            expected_aspects = await self._generate_expected_aspects(kp_name)

        if not expected_aspects:
            return {"aspects": [], "completeness": 0.0}

        aspects_text = "\n".join(f"- {a}" for a in expected_aspects)
        prompt = JUDGE_COMPLETENESS_PROMPT.format(
            kp_name=kp_name,
            expected_aspects=aspects_text,
            generated_content=generated_content[:3000],
        )
        try:
            raw = await chat_completion(
                [{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=1000,
                provider=self.provider,
            )
            result = safe_json_loads(raw)
            n_covered = sum(
                1 for a in result.get("aspects", [])
                if a.get("coverage") == "covered"
            )
            logger.info(
                f"[RAGJudge] completeness={result.get('completeness', 0):.2f} "
                f"aspects={len(result.get('aspects', []))} covered={n_covered}"
            )
            return result
        except Exception as e:
            logger.warning(f"[RAGJudge] completeness 评估失败: {e}")
            return {"aspects": [], "completeness": 0.0}

    # ----------------------------------------------------------
    # Judge 4: Citation Accuracy
    # ----------------------------------------------------------

    async def judge_citation_accuracy(
        self,
        citation_context: str,
        reference_chunk: str,
        ref_index: int = 1,
    ) -> dict:
        """
        评估单条引用的准确性。

        :param citation_context: 引用处的上下文（前后各 200 字）
        :param reference_chunk:  引用指向的参考资料 chunk 内容
        :param ref_index:        引用编号（如 [1] 中的 1）
        :return:                 {"verdict": "accurate|inaccurate|vague", "explanation": "..."}
        """
        prompt = JUDGE_CITATION_PROMPT.format(
            ref_index=ref_index,
            citation_context=citation_context[:1500],
            reference_chunk=reference_chunk[:1500],
        )
        try:
            raw = await chat_completion(
                [{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=300,
                provider=self.provider,
            )
            result = safe_json_loads(raw)
            logger.debug(
                f"[RAGJudge] citation_accuracy ref=[{ref_index}] "
                f"verdict={result.get('verdict', '?')}"
            )
            return result
        except Exception as e:
            logger.warning(f"[RAGJudge] citation_accuracy 评估失败: {e}")
            return {"verdict": "vague", "explanation": f"评估异常: {e}"}

    async def judge_citation_accuracy_batch(
        self,
        generated_content: str,
        retrieved_chunks: list,
    ) -> dict:
        """
        批量评估生成内容中所有引用标注的准确性。

        策略：从 generated_content 中提取所有 [n] 引用 → 找到对应 chunk → 逐条评估。

        :param generated_content: LLM 生成的答案（含 [n] 引用标注）
        :param retrieved_chunks:  RetrievedChunk 列表
        :return:                  {"citations": [...], "citation_precision": 0.0-1.0}
        """
        import asyncio

        # 提取所有引用标注 [1], [2], ...
        citation_refs = re.findall(r'\[(\d+)\]', generated_content)
        unique_refs = sorted(set(int(r) for r in citation_refs))

        if not unique_refs or not retrieved_chunks:
            return {"citations": [], "citation_precision": 1.0}

        async def _eval_one(ref_idx: int) -> dict:
            # 找到引用附近的上下文
            pattern = re.compile(rf'\[{ref_idx}\]')
            match = pattern.search(generated_content)
            if not match:
                return {"ref_index": ref_idx, "verdict": "vague", "explanation": "未找到引用位置"}
            start = max(0, match.start() - 200)
            end = min(len(generated_content), match.end() + 200)
            context = generated_content[start:end]

            # 找到对应的 chunk（1-indexed → 0-indexed）
            if ref_idx <= len(retrieved_chunks):
                chunk = retrieved_chunks[ref_idx - 1]
                chunk_text = chunk.text if hasattr(chunk, "text") else str(chunk)
            else:
                return {"ref_index": ref_idx, "verdict": "vague", "explanation": "引用编号超出参考资料范围"}

            result = await self.judge_citation_accuracy(context, chunk_text, ref_idx)
            result["ref_index"] = ref_idx
            return result

        tasks = [_eval_one(r) for r in unique_refs[:10]]  # 最多评估 10 条引用
        results = await asyncio.gather(*tasks, return_exceptions=True)

        citations = []
        accurate_count = 0
        for r in results:
            if isinstance(r, dict):
                citations.append(r)
                if r.get("verdict") == "accurate":
                    accurate_count += 1

        precision = accurate_count / len(citations) if citations else 1.0
        return {"citations": citations, "citation_precision": round(precision, 4)}

    # ----------------------------------------------------------
    # 联合评估
    # ----------------------------------------------------------

    async def evaluate_full(
        self,
        query: str,
        kp_name: str,
        retrieved_chunks: list,
        generated_content: str,
        include_citation_check: bool = True,
        experiment_group: str | None = None,
        cross_validate: bool | None = None,
        safety_issues: list[str] | None = None,
    ) -> dict:
        """
        执行完整的四维度 LLM-as-Judge 评估。

        :param query:                  用户查询
        :param kp_name:                知识点名称
        :param retrieved_chunks:       RetrievedChunk 列表
        :param generated_content:      LLM 生成答案
        :param include_citation_check: 是否包含引用准确性评估
        :param experiment_group:       A/B 实验分组标签
        :param cross_validate:         是否启用多 LLM 交叉验证，None=使用 config 设置
        :param safety_issues:          SafetyAgent 已发现的问题列表（可选，辅助 Faithfulness Judge）
        :return:                       完整评估结果 dict
        """
        t_start = time.perf_counter()
        logger.info(f"[RAGJudge] 开始完整评估: kp_name={kp_name}, n_chunks={len(retrieved_chunks)}, content_len={len(generated_content)}")

        retrieved_texts = [
            c.text if hasattr(c, "text") else str(c)
            for c in retrieved_chunks
        ]

        # 并行执行 Judge 1 + 2 + 3
        import asyncio

        relevance_task = asyncio.create_task(
            self.judge_chunk_relevance_batch(query, retrieved_chunks)
        )
        faithfulness_task = asyncio.create_task(
            self.judge_faithfulness(retrieved_texts, generated_content, safety_issues)
        )
        completeness_task = asyncio.create_task(
            self.judge_completeness(kp_name, generated_content)
        )

        relevance_labels, faithfulness_result, completeness_result = await asyncio.gather(
            relevance_task, faithfulness_task, completeness_task,
        )

        # Judge 4: Citation Accuracy（可选，仅在生成内容包含引用标注时执行）
        citation_result = None
        if include_citation_check and re.search(r'\[\d+\]', generated_content):
            citation_result = await self.judge_citation_accuracy_batch(
                generated_content, retrieved_chunks
            )

        # 聚合结果
        faithfulness_score = faithfulness_result.get("faithfulness", 0.0)
        statements = faithfulness_result.get("statements", [])
        hallucination = (
            1.0 - faithfulness_score
            if statements
            else 0.0
        )

        # 多 LLM 交叉验证（可选，开发阶段调试用）
        cross_validated = False
        cross_validation_disagreement = False
        cv_providers = None
        if cross_validate is None:
            cross_validate = config.evaluation.cross_validation.enabled
        if cross_validate:
            cv_providers = config.evaluation.cross_validation.providers
            if len(cv_providers) >= 2:
                cv_result = await self._cross_validate(
                    query, kp_name, retrieved_chunks, generated_content,
                    primary_provider=self.provider,
                    secondary_provider=cv_providers[1] if cv_providers[1] != self.provider else cv_providers[0],
                    primary_faithfulness=faithfulness_score,
                    primary_completeness=completeness_result.get("completeness", 0.0),
                )
                cross_validated = True
                cross_validation_disagreement = cv_result.get("disagreement", False)
                if cross_validation_disagreement:
                    logger.warning(
                        f"[RAGJudge] 交叉验证不一致！primary vs secondary 评分差异显著"
                    )

        elapsed_ms = (time.perf_counter() - t_start) * 1000

        result = {
            "query": query,
            "kp_name": kp_name,
            "timestamp": None,  # 由外部设置
            # Judge 1: 检索相关性
            "relevance_labels": relevance_labels,
            "precision_at_5": precision_at_k(relevance_labels, 5) if relevance_labels else 0.0,
            # Judge 2: 忠实度
            "faithfulness_score": faithfulness_score,
            "hallucination_rate": hallucination,
            "faithfulness_statements": statements,
            "faithfulness_issues": faithfulness_result.get("issues", []),
            # Judge 3: 完整度
            "completeness_score": completeness_result.get("completeness", 0.0),
            "completeness_aspects": completeness_result.get("aspects", []),
            # Judge 4: 引用准确性
            "citation_precision": citation_result.get("citation_precision") if citation_result else None,
            "citations": citation_result.get("citations", []) if citation_result else [],
            # 多 LLM 交叉验证
            "cross_validated": cross_validated,
            "cross_validation_disagreement": cross_validation_disagreement,
            # SafetyAgent 审核结论（透传）
            "safety_issues": safety_issues or [],
            # 元数据
            "experiment_group": experiment_group,
            "evaluation_time_ms": round(elapsed_ms, 1),
        }

        summary_parts = [
            f"faithfulness={faithfulness_score:.2f}",
            f"completeness={result['completeness_score']:.2f}",
            f"P@5={result['precision_at_5']:.2f}",
            f"hallucination={hallucination:.2f}",
        ]
        if citation_result:
            summary_parts.append(f"citation_precision={citation_result.get('citation_precision', 0):.2f}")
        summary_parts.append(f"elapsed={elapsed_ms:.0f}ms")

        logger.info(f"[RAGJudge] 评估完成: {', '.join(summary_parts)}")

        # 如果幻觉率高，发出醒目告警
        if hallucination > 0.3:
            logger.warning(
                f"[RAGJudge] 幻觉率偏高 ({hallucination:.1%})！"
                f"issues={faithfulness_result.get('issues', [])}"
            )
        if faithfulness_score < 0.5:
            logger.warning(
                f"[RAGJudge] 忠实度过低 ({faithfulness_score:.1%})，"
                f"可能存在严重的参考资料脱节"
            )

        return result

    # ----------------------------------------------------------
    # 多 LLM 交叉验证
    # ----------------------------------------------------------

    async def _cross_validate(
        self,
        query: str,
        kp_name: str,
        retrieved_chunks: list,
        generated_content: str,
        primary_provider: str,
        secondary_provider: str,
        primary_faithfulness: float,
        primary_completeness: float,
    ) -> dict:
        """
        用第二个 LLM 对关键维度（忠实度 + 完整度）重新打分，检测评分偏差。

        当两个 LLM 的忠实度或完整度评分相差 >0.3 时，标记为 disagreement。
        """
        logger.info(
            f"[RAGJudge] 交叉验证: primary={primary_provider}, secondary={secondary_provider}"
        )
        try:
            retrieved_texts = [
                c.text if hasattr(c, "text") else str(c)
                for c in retrieved_chunks
            ]

            # 仅用第二个 provider 重新评估忠实度和完整度（最大开销项）
            original_provider = self.provider
            self.provider = secondary_provider

            import asyncio
            try:
                cv_faithfulness_task = asyncio.create_task(
                    self.judge_faithfulness(retrieved_texts, generated_content)
                )
                cv_completeness_task = asyncio.create_task(
                    self.judge_completeness(kp_name, generated_content)
                )
                cv_faithfulness, cv_completeness = await asyncio.gather(
                    cv_faithfulness_task, cv_completeness_task,
                )
            finally:
                self.provider = original_provider

            cv_faith = cv_faithfulness.get("faithfulness", 0.0)
            cv_comp = cv_completeness.get("completeness", 0.0)

            disagreement = (
                abs(primary_faithfulness - cv_faith) > 0.3
                or abs(primary_completeness - cv_comp) > 0.3
            )

            logger.info(
                f"[RAGJudge] 交叉验证结果: "
                f"faithfulness primary={primary_faithfulness:.2f} vs secondary={cv_faith:.2f} "
                f"(delta={abs(primary_faithfulness - cv_faith):.2f}), "
                f"completeness primary={primary_completeness:.2f} vs secondary={cv_comp:.2f} "
                f"(delta={abs(primary_completeness - cv_comp):.2f}), "
                f"disagreement={disagreement}"
            )

            return {
                "primary_provider": primary_provider,
                "secondary_provider": secondary_provider,
                "faithfulness_secondary": cv_faith,
                "completeness_secondary": cv_comp,
                "disagreement": disagreement,
            }
        except Exception as e:
            logger.warning(f"[RAGJudge] 交叉验证失败: {e}")
            return {"disagreement": False}


# 模块级单例
_judge_instance: RAGJudge | None = None


def get_judge(
    provider: str | None = None,
    temperature: float = 0.0,
    sample_rate: float = 0.1,
) -> RAGJudge:
    """获取 RAGJudge 模块级单例。"""
    global _judge_instance
    if _judge_instance is None:
        _judge_instance = RAGJudge(
            provider=provider,
            temperature=temperature,
            sample_rate=sample_rate,
        )
    return _judge_instance
