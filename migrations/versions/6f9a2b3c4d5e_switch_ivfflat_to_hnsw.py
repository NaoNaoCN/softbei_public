"""switch embedding index from IVFFlat to HNSW

IVFFlat 需要定期 REINDEX 才能覆盖增量插入的向量，与项目"按文档增量更新"的
写入模式不兼容。HNSW 支持插入即索引，无需维护。

Revision ID: 6f9a2b3c4d5e
Revises: 5d8e1f2a3b4c
Create Date: 2026-05-25
"""
from typing import Sequence, Union

from alembic import op

revision: str = '6f9a2b3c4d5e'
down_revision: Union[str, None] = '5d8e1f2a3b4c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. 删除旧 IVFFlat 索引
    op.execute("DROP INDEX IF EXISTS ix_document_chunk_embedding_ivfflat")

    # 2. 创建 HNSW 索引
    #    m=16: 每个节点最大邻居数，精度/内存/速度的平衡点
    #    ef_construction=200: 构建时搜索宽度，控制索引质量
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_document_chunk_embedding_hnsw
        ON document_chunk
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 200)
    """)


def downgrade() -> None:
    # 回退到 IVFFlat
    op.execute("DROP INDEX IF EXISTS ix_document_chunk_embedding_hnsw")
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_document_chunk_embedding_ivfflat
        ON document_chunk
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100)
    """)
