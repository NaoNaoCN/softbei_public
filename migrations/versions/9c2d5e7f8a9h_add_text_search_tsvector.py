"""add text_search tsvector column with GIN index for full-text search

在 document_chunk 表添加 text_search 列 (tsvector) 和 GIN 索引，
支持基于 jieba 分词 + PostgreSQL 全文检索的关键词召回路径。
替代原有的 ILIKE 子串匹配方案。

Revision ID: 9c2d5e7f8a9h
Revises: 8b3c4d5e6f7g
Create Date: 2026-05-27
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = '9c2d5e7f8a9h'
down_revision: Union[str, None] = '8b3c4d5e6f7g'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'document_chunk',
        sa.Column(
            'text_search',
            postgresql.TSVECTOR,
            nullable=True,
        ),
    )
    op.create_index(
        'ix_document_chunk_text_search',
        'document_chunk',
        ['text_search'],
        unique=False,
        postgresql_using='gin',
    )


def downgrade() -> None:
    op.drop_index('ix_document_chunk_text_search', table_name='document_chunk')
    op.drop_column('document_chunk', 'text_search')
