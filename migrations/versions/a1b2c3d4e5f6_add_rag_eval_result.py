"""add rag_eval_result table for persistent evaluation storage

Revision ID: a1b2c3d4e5f6
Revises: 9c2d5e7f8a9h
Create Date: 2026-05-27
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, None] = '9c2d5e7f8a9h'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'rag_eval_result',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('session_id', sa.String(64), server_default=''),
        sa.Column('user_id', sa.String(64), server_default=''),
        sa.Column('agent_type', sa.String(32), server_default=''),
        sa.Column('kp_name', sa.String(512), server_default=''),
        sa.Column('query', sa.Text(), server_default=''),
        sa.Column('experiment_group', sa.String(64), nullable=True),

        # 检索指标
        sa.Column('n_retrieved', sa.Integer(), server_default='0'),
        sa.Column('n_candidates', sa.Integer(), server_default='0'),
        sa.Column('scores_min', sa.Float(), nullable=True),
        sa.Column('scores_p50', sa.Float(), nullable=True),
        sa.Column('scores_max', sa.Float(), nullable=True),
        sa.Column('embedding_latency_ms', sa.Float(), server_default='0.0'),
        sa.Column('db_query_latency_ms', sa.Float(), server_default='0.0'),

        # Judge 评分
        sa.Column('relevance_labels', JSONB, nullable=True),
        sa.Column('precision_at_5', sa.Float(), nullable=True),
        sa.Column('recall_at_5', sa.Float(), nullable=True),
        sa.Column('ndcg_at_5', sa.Float(), nullable=True),
        sa.Column('faithfulness_score', sa.Float(), nullable=True),
        sa.Column('hallucination_rate', sa.Float(), nullable=True),
        sa.Column('completeness_score', sa.Float(), nullable=True),
        sa.Column('citation_precision', sa.Float(), nullable=True),

        # 交叉验证
        sa.Column('cross_validated', sa.Boolean(), server_default='false'),
        sa.Column('cross_validation_disagreement', sa.Boolean(), server_default='false'),

        # 元数据
        sa.Column('extra_metadata', JSONB, nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),

        sa.PrimaryKeyConstraint('id'),
    )

    # 索引
    op.create_index('ix_rag_eval_result_session_id', 'rag_eval_result', ['session_id'])
    op.create_index('ix_rag_eval_result_agent_type', 'rag_eval_result', ['agent_type'])
    op.create_index('ix_rag_eval_result_created_at', 'rag_eval_result', ['created_at'])
    op.create_index('ix_rag_eval_result_experiment_group', 'rag_eval_result', ['experiment_group'])


def downgrade() -> None:
    op.drop_table('rag_eval_result')
