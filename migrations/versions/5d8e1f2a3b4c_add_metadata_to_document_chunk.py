"""add_metadata_to_document_chunk

Revision ID: 5d8e1f2a3b4c
Revises: 107bc3a0d271
Create Date: 2026-05-21 18:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


# revision identifiers, used by Alembic.
revision: str = '5d8e1f2a3b4c'
down_revision: Union[str, Sequence[str], None] = '107bc3a0d271'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """为 document_chunk 添加 JSONB 元数据列及 GIN 索引。"""
    op.add_column(
        'document_chunk',
        sa.Column('metadata', JSONB, nullable=True),
    )
    # GIN 索引加速 JSONB 内字段过滤 (metadata->>'key')
    op.create_index(
        'ix_document_chunk_metadata',
        'document_chunk',
        ['metadata'],
        postgresql_using='gin',
    )


def downgrade() -> None:
    """移除 metadata 列及索引。"""
    op.drop_index(
        'ix_document_chunk_metadata',
        table_name='document_chunk',
    )
    op.drop_column('document_chunk', 'metadata')
