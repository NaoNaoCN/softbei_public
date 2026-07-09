"""add parent_chunk_id and is_parent to document_chunk for parent-child chunking

在 document_chunk 表添加 parent_chunk_id (VARCHAR) 和 is_parent (BOOLEAN) 列，
支持父子切割检索模式：子块参与向量检索，检索命中后回填父块文本以提供更完整的上下文。

Revision ID: 8b3c4d5e6f7g
Revises: 7a1b2c3d4e5f
Create Date: 2026-05-25
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '8b3c4d5e6f7g'
down_revision: Union[str, None] = '6f9a2b3c4d5e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. 添加 parent_chunk_id 列 — 子块指向父块的 chunk_id
    op.add_column(
        'document_chunk',
        sa.Column('parent_chunk_id', sa.String(128), nullable=True),
    )

    # 2. 添加 is_parent 列 — 标记自身是否为父块
    op.add_column(
        'document_chunk',
        sa.Column('is_parent', sa.Boolean, nullable=False, server_default=sa.text('FALSE')),
    )

    # 3. 创建 parent_chunk_id 索引，加速父块回填查询
    op.create_index(
        'ix_document_chunk_parent',
        'document_chunk',
        ['parent_chunk_id'],
    )

    # 4. 创建 is_parent 索引，加速父块批量查询
    op.create_index(
        'ix_document_chunk_is_parent',
        'document_chunk',
        ['is_parent'],
    )


def downgrade() -> None:
    op.drop_index('ix_document_chunk_is_parent', table_name='document_chunk')
    op.drop_index('ix_document_chunk_parent', table_name='document_chunk')
    op.drop_column('document_chunk', 'is_parent')
    op.drop_column('document_chunk', 'parent_chunk_id')
