"""add study_plan and study_plan_item tables

Revision ID: c3d4e5f6a7b8
Revises: 13c01e0c4c91
Create Date: 2026-06-04
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'c3d4e5f6a7b8'
down_revision: Union[str, None] = '13c01e0c4c91'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'study_plan',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('user_id', sa.BigInteger(), nullable=False),
        sa.Column('title', sa.String(256), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('goal', sa.Text(), nullable=True),
        sa.Column('start_date', sa.Date(), nullable=False),
        sa.Column('end_date', sa.Date(), nullable=False),
        sa.Column('daily_time_minutes', sa.Integer(), nullable=True),
        sa.Column('status', sa.String(16), server_default='active', nullable=False),
        sa.Column('source_path_ids', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.now()),
        sa.ForeignKeyConstraint(['user_id'], ['user.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_study_plan_user_id', 'study_plan', ['user_id'])
    op.create_index('ix_study_plan_user_status', 'study_plan', ['user_id', 'status'])

    op.create_table(
        'study_plan_item',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('plan_id', sa.BigInteger(), nullable=False),
        sa.Column('kp_id', sa.String(64), nullable=True),
        sa.Column('kp_name', sa.String(256), nullable=False),
        sa.Column('scheduled_date', sa.Date(), nullable=False),
        sa.Column('start_time', sa.String(5), nullable=True),
        sa.Column('end_time', sa.String(5), nullable=True),
        sa.Column('estimated_minutes', sa.Integer(), nullable=True),
        sa.Column('order_index', sa.Integer(), server_default='0'),
        sa.Column('is_completed', sa.Boolean(), server_default=sa.false()),
        sa.Column('resource_ids', sa.JSON(), nullable=True),
        sa.Column('missing_resource_types', sa.JSON(), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(['plan_id'], ['study_plan.id']),
        sa.ForeignKeyConstraint(['kp_id'], ['kg_node.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_study_plan_item_plan_id', 'study_plan_item', ['plan_id'])
    op.create_index('ix_study_plan_item_plan_date', 'study_plan_item', ['plan_id', 'scheduled_date'])


def downgrade() -> None:
    op.drop_table('study_plan_item')
    op.drop_table('study_plan')
