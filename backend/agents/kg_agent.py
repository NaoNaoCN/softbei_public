"""
backend/agents/kg_agent.py
KGAgent：从已导入文档自动构建知识图谱（KGNode + KGEdge）。
"""

from __future__ import annotations

from loguru import logger

from langchain_core.runnables import RunnableConfig

from backend.models.schemas import AgentState
from backend.services.kg_builder import build_kg

async def run(state: AgentState, config: RunnableConfig = None) -> AgentState:
    """
    KGAgent 节点入口。

    从 state.kg_doc_id 或 state.metadata["doc_id"] 获取目标文档 ID，
    调用 kg_builder 构建知识图谱，将结果写入 state。
    """
    # 获取 db
    db = None
    if config and "configurable" in config:
        db = config["configurable"].get("db")

    if not db:
        state = state.model_copy(update={
            "final_content": "知识图谱构建失败：无法获取数据库连接",
            "error": "no db session",
        })
        return state

    # 获取 doc_id
    doc_id = state.kg_doc_id or state.metadata.get("doc_id")
    if not doc_id:
        # 尝试从 kp_id 推断（用户可能说了文档名）
        state = state.model_copy(update={
            "final_content": "请指定要构建知识图谱的文档。您可以在资源库页面选择已导入的 PDF 文档，点击「构建知识图谱」按钮。",
        })
        return state

    try:
        result = await build_kg(doc_id, db)
        nodes_count = result["nodes_count"]
        edges_count = result["edges_count"]

        report = (
            f"知识图谱构建完成！\n\n"
            f"- 文档 ID: {doc_id}\n"
            f"- 提取知识点: {nodes_count} 个\n"
            f"- 推断关系: {edges_count} 条\n\n"
            f"您可以前往「学习路径」页面查看知识图谱可视化。"
        )
        state = state.model_copy(update={
            "final_content": report,
            "draft_content": report,
            "metadata": {**state.metadata, "kg_result": result},
        })
    except ValueError as e:
        state = state.model_copy(update={
            "final_content": f"知识图谱构建失败：{e}",
            "error": str(e),
        })
    except Exception as e:
        logger.exception(f"[KGAgent] 构建失败: {e}")
        state = state.model_copy(update={
            "final_content": f"知识图谱构建过程中出错：{e}",
            "error": str(e),
        })

    return state
