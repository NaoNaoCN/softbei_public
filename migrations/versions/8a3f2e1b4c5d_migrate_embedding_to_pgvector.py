"""migrate embedding column from JSON to pgvector

Revision ID: 8a3f2e1b4c5d
Revises: db2c961ff39d
Create Date: 2026-05-18
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

revision: str = '8a3f2e1b4c5d'
down_revision: Union[str, None] = 'db2c961ff39d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. 启用 pgvector 扩展
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # 2. 删除旧 JSON 列（当前表为空，无需数据迁移）
    op.drop_column("document_chunk", "embedding")

    # 3. 新建 vector(1024) 列
    op.add_column(
        "document_chunk",
        sa.Column("embedding", Vector(1024), nullable=True),
    )

    # 4. 创建 IVFFlat 索引（lists=100 适合万级数据）
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_document_chunk_embedding_ivfflat
        ON document_chunk
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100)
    """)


def downgrade() -> None:
    op.drop_index(
        "ix_document_chunk_embedding_ivfflat",
        table_name="document_chunk",
        if_exists=True,
    )
    op.drop_column("document_chunk", "embedding")
    op.add_column(
        "document_chunk",
        sa.Column("embedding", sa.JSON(), nullable=True),
    )
