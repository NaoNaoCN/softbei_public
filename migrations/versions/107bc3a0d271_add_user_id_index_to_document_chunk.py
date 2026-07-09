"""add_user_id_index_to_document_chunk

Revision ID: 107bc3a0d271
Revises: 8a3f2e1b4c5d
Create Date: 2026-05-20 14:59:55.442141

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '107bc3a0d271'
down_revision: Union[str, Sequence[str], None] = '8a3f2e1b4c5d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """为 document_chunk.user_id 添加索引，加速多用户检索过滤。"""
    op.create_index(
        'ix_document_chunk_user_id',
        'document_chunk',
        ['user_id'],
    )


def downgrade() -> None:
    """移除 user_id 索引。"""
    op.drop_index(
        'ix_document_chunk_user_id',
        table_name='document_chunk',
    )
