"""add rag_eval_analysis table for evaluation analysis results

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-05-27
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


revision: str = 'b2c3d4e5f6a7'
down_revision: Union[str, None] = 'a1b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'rag_eval_analysis',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('analysis_type', sa.String(32), server_default='golden_run'),
        sa.Column('source_reference', sa.String(512), server_default=''),

        # Aggregate scores
        sa.Column('avg_faithfulness', sa.Float(), server_default='0.0'),
        sa.Column('avg_completeness', sa.Float(), server_default='0.0'),
        sa.Column('avg_precision_at_5', sa.Float(), server_default='0.0'),
        sa.Column('pass_rate', sa.Float(), server_default='0.0'),
        sa.Column('total_queries', sa.Integer(), server_default='0'),

        # Analysis results (JSONB)
        sa.Column('bottlenecks', JSONB, nullable=True),
        sa.Column('strengths', JSONB, nullable=True),
        sa.Column('weaknesses', JSONB, nullable=True),
        sa.Column('suggestions', JSONB, nullable=True),
        sa.Column('key_findings', JSONB, nullable=True),
        sa.Column('per_kp_breakdown', JSONB, nullable=True),
        sa.Column('trend', sa.String(32), server_default='stable'),
        sa.Column('extra_metadata', JSONB, nullable=True),

        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint('id'),
    )

    op.create_index('ix_rag_eval_analysis_type', 'rag_eval_analysis', ['analysis_type'])
    op.create_index('ix_rag_eval_analysis_created_at', 'rag_eval_analysis', ['created_at'])


def downgrade() -> None:
    op.drop_table('rag_eval_analysis')
