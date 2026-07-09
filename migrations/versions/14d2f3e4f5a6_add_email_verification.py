"""add email fields to user and email_verification table

Revision ID: 14d2f3e4f5a6
Revises: 13c01e0c4c91
Create Date: 2026-06-05 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '14d2f3e4f5a6'
down_revision: Union[str, Sequence[str], None] = '13c01e0c4c91'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add email columns to user table
    op.add_column('user', sa.Column('email', sa.String(256), nullable=True, unique=True))
    op.add_column('user', sa.Column('email_verified', sa.Boolean(), server_default=sa.text('false'), nullable=False))
    op.add_column('user', sa.Column('email_verified_at', sa.DateTime(), nullable=True))

    # Create email_verification table
    op.create_table(
        'email_verification',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('user_id', sa.BigInteger(), nullable=False),
        sa.Column('token_hash', sa.String(128), nullable=False),
        sa.Column('purpose', sa.String(32), nullable=False),
        sa.Column('expires_at', sa.DateTime(), nullable=False),
        sa.Column('used', sa.Boolean(), server_default=sa.text('false'), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['user.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_email_verification_token_hash', 'email_verification', ['token_hash'])


def downgrade() -> None:
    op.drop_table('email_verification')
    op.drop_column('user', 'email_verified_at')
    op.drop_column('user', 'email_verified')
    op.drop_column('user', 'email')
