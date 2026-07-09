"""merge email_verification and study_plan heads

Revision ID: 077c06b272a1
Revises: 14d2f3e4f5a6, c3d4e5f6a7b8
Create Date: 2026-06-05 22:55:55.876731

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '077c06b272a1'
down_revision: Union[str, Sequence[str], None] = ('14d2f3e4f5a6', 'c3d4e5f6a7b8')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
