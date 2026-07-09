"""uuid_to_snowflake

将项目所有主键/外键从 PostgreSQL UUID 类型改为 BIGINT（Snowflake ID）。

=============================================================================
  ** 严重警告 / CRITICAL WARNING **
  本迁移会 DROP 并重建全部 17 张表，现有数据将永久丢失且无法恢复。
  This migration DROPs and recreates ALL 17 tables. All existing data will be
  PERMANENTLY LOST with no recovery path.

  切勿在生产环境或有数据的测试环境运行。
  DO NOT run against any environment containing real data.

  如需要在保留数据的前提下切换主键类型，方案如下：
  1. pg_dump 导出全部数据
  2. 创建新库并运行迁移链（从 head 开始直达此迁移）
  3. 编写脚本将旧数据转换并导入新库
  4. 切换应用连接到新库
=============================================================================

开发环境执行：
    alembic upgrade head

Revision ID: db2c961ff39d
Revises: 9069c06f0251
Create Date: 2026-05-15
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'db2c961ff39d'
down_revision: Union[str, Sequence[str], None] = '9069c06f0251'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """将所有 UUID 列改为 BIGINT — 按 FK 依赖逆序删表，再按正序重建。"""
    # ---- 阶段 1：按 FK 依赖逆序删除（子表先删，父表后删） ----
    op.drop_table('learning_record')
    op.drop_table('learning_path_item')
    op.drop_table('learning_path')
    op.drop_table('quiz_attempt')
    op.drop_table('quiz_item')
    op.drop_table('generation_task')
    op.drop_table('generation_batch')
    op.drop_table('resource_meta')
    op.drop_table('profile_history')
    op.drop_table('student_profile')
    op.drop_table('chat_message')
    op.drop_table('chat_session')
    op.drop_table('kg_edge')
    op.drop_table('kg_build_task')
    op.drop_table('kg_node')
    op.drop_table('document_chunk')
    op.drop_table('user')

    # ---- 阶段 2：按 FK 依赖正序重建（父表先建，子表后建） ----
    _create_tables_bigint()


def downgrade() -> None:
    """回退到 UUID — 同样重建所有表。"""
    op.drop_table('learning_record')
    op.drop_table('learning_path_item')
    op.drop_table('learning_path')
    op.drop_table('quiz_attempt')
    op.drop_table('quiz_item')
    op.drop_table('generation_task')
    op.drop_table('generation_batch')
    op.drop_table('resource_meta')
    op.drop_table('profile_history')
    op.drop_table('student_profile')
    op.drop_table('chat_message')
    op.drop_table('chat_session')
    op.drop_table('kg_edge')
    op.drop_table('kg_build_task')
    op.drop_table('kg_node')
    op.drop_table('document_chunk')
    op.drop_table('user')

    _create_tables_uuid()


def _create_tables_bigint() -> None:
    """用 BIGINT 类型重建所有表。"""
    # 1. user（根表，无外键依赖）
    op.create_table('user',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('username', sa.String(64), nullable=False),
        sa.Column('hashed_password', sa.String(128), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('username'),
    )

    # 2. document_chunk（无 FK 依赖除 user_id 外）
    op.create_table('document_chunk',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('chunk_id', sa.String(128), nullable=False),
        sa.Column('doc_id', sa.String(128), nullable=False),
        sa.Column('collection_name', sa.String(64), nullable=False, server_default='knowledge_base'),
        sa.Column('text', sa.Text(), nullable=False),
        sa.Column('embedding', sa.JSON(), nullable=True),
        sa.Column('source', sa.String(512), nullable=True),
        sa.Column('page', sa.Integer(), nullable=True),
        sa.Column('section', sa.String(256), nullable=True),
        sa.Column('user_id', sa.String(64), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('chunk_id'),
    )
    op.create_index('ix_document_chunk_chunk_id', 'document_chunk', ['chunk_id'])
    op.create_index('ix_document_chunk_doc_id', 'document_chunk', ['doc_id'])
    op.create_index('ix_document_chunk_collection_name', 'document_chunk', ['collection_name'])

    # 3. kg_node（FK → user）
    op.create_table('kg_node',
        sa.Column('id', sa.String(64), nullable=False),
        sa.Column('name', sa.String(256), nullable=False),
        sa.Column('node_type', sa.String(32), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('course_id', sa.String(64), nullable=True),
        sa.Column('user_id', sa.BigInteger(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['user.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_kg_node_user_id', 'kg_node', ['user_id'])
    op.create_index('ix_kg_node_user_type', 'kg_node', ['user_id', 'node_type'])

    # 4. chat_session（FK → user）
    op.create_table('chat_session',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('user_id', sa.BigInteger(), nullable=False),
        sa.Column('title', sa.String(256), nullable=True),
        sa.Column('last_used_at', sa.DateTime(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['user.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_chat_session_user_id', 'chat_session', ['user_id'])

    # 5. student_profile（FK → user，user_id 已通过 UNIQUE 自动索引）
    op.create_table('student_profile',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('user_id', sa.BigInteger(), nullable=False),
        sa.Column('major', sa.String(128), nullable=True),
        sa.Column('learning_goal', sa.Text(), nullable=True),
        sa.Column('cognitive_style', sa.String(32), nullable=True),
        sa.Column('daily_time_minutes', sa.Integer(), nullable=True),
        sa.Column('knowledge_mastered', sa.JSON(), nullable=True),
        sa.Column('knowledge_weak', sa.JSON(), nullable=True),
        sa.Column('error_prone', sa.JSON(), nullable=True),
        sa.Column('current_progress', sa.Text(), nullable=True),
        sa.Column('goal_questions', sa.JSON(), nullable=True),
        sa.Column('version', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['user.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('user_id'),
    )

    # 6. resource_meta（FK → user，高频查询表）
    op.create_table('resource_meta',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('user_id', sa.BigInteger(), nullable=False),
        sa.Column('kp_id', sa.String(256), nullable=False),
        sa.Column('resource_type', sa.String(32), nullable=False),
        sa.Column('title', sa.String(256), nullable=True),
        sa.Column('content', sa.Text(), nullable=True),
        sa.Column('content_json', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['user.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_resource_meta_user_id', 'resource_meta', ['user_id'])
    op.create_index('ix_resource_meta_kp_id', 'resource_meta', ['kp_id'])
    op.create_index('ix_resource_meta_user_type', 'resource_meta', ['user_id', 'resource_type'])
    op.create_index('ix_resource_meta_user_kp', 'resource_meta', ['user_id', 'kp_id'])

    # 7. learning_path（FK → user）
    op.create_table('learning_path',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('user_id', sa.BigInteger(), nullable=False),
        sa.Column('title', sa.String(256), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['user.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_learning_path_user_id', 'learning_path', ['user_id'])

    # 8. kg_build_task（FK → user）
    op.create_table('kg_build_task',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('doc_id', sa.String(128), nullable=False),
        sa.Column('user_id', sa.BigInteger(), nullable=True),
        sa.Column('status', sa.String(32), nullable=False, server_default='pending'),
        sa.Column('progress', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('stage', sa.String(64), nullable=True),
        sa.Column('nodes_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('edges_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['user.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_kg_build_task_user_id', 'kg_build_task', ['user_id'])
    op.create_index('ix_kg_build_task_doc_id', 'kg_build_task', ['doc_id'])

    # 9. generation_batch（FK → user）
    op.create_table('generation_batch',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('user_id', sa.BigInteger(), nullable=False),
        sa.Column('kp_id', sa.String(256), nullable=False),
        sa.Column('status', sa.String(32), nullable=False, server_default='pending'),
        sa.Column('progress', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('resource_types', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['user.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_generation_batch_user_id', 'generation_batch', ['user_id'])

    # ---- 子孙表（依赖上面的父表） ----

    # 10. chat_message（FK → chat_session）
    op.create_table('chat_message',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('session_id', sa.BigInteger(), nullable=False),
        sa.Column('role', sa.String(16), nullable=False),
        sa.Column('content', sa.Text(), nullable=True),
        sa.Column('resource_type', sa.String(16), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['session_id'], ['chat_session.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_chat_message_session_time', 'chat_message', ['session_id', 'created_at'])
    op.create_index('ix_chat_message_session_id', 'chat_message', ['session_id'])

    # 11. profile_history（FK → student_profile）
    op.create_table('profile_history',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('profile_id', sa.BigInteger(), nullable=False),
        sa.Column('snapshot', sa.JSON(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['profile_id'], ['student_profile.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_profile_history_profile_id', 'profile_history', ['profile_id'])

    # 12. kg_edge（FK → kg_node ×2）
    op.create_table('kg_edge',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('source_id', sa.String(64), nullable=False),
        sa.Column('target_id', sa.String(64), nullable=False),
        sa.Column('relation', sa.String(32), nullable=False),
        sa.ForeignKeyConstraint(['source_id'], ['kg_node.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['target_id'], ['kg_node.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('source_id', 'target_id', 'relation'),
    )

    # 13. generation_task（FK → resource_meta + generation_batch）
    op.create_table('generation_task',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('resource_id', sa.BigInteger(), nullable=False),
        sa.Column('batch_id', sa.BigInteger(), nullable=True),
        sa.Column('status', sa.String(32), nullable=False, server_default='pending'),
        sa.Column('progress', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['resource_id'], ['resource_meta.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['batch_id'], ['generation_batch.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('resource_id'),
    )
    op.create_index('ix_generation_task_batch_id', 'generation_task', ['batch_id'])

    # 14. quiz_item（FK → resource_meta）
    op.create_table('quiz_item',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('resource_id', sa.BigInteger(), nullable=False),
        sa.Column('kp_id', sa.String(64), nullable=True),
        sa.Column('question_type', sa.String(16), nullable=False),
        sa.Column('stem', sa.Text(), nullable=False),
        sa.Column('options', sa.JSON(), nullable=True),
        sa.Column('answer', sa.Text(), nullable=False),
        sa.Column('explanation', sa.Text(), nullable=True),
        sa.Column('order_index', sa.Integer(), nullable=False, server_default='0'),
        sa.ForeignKeyConstraint(['resource_id'], ['resource_meta.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_quiz_item_resource_id', 'quiz_item', ['resource_id'])

    # 15. quiz_attempt（FK → quiz_item + user）
    op.create_table('quiz_attempt',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('quiz_item_id', sa.BigInteger(), nullable=False),
        sa.Column('user_id', sa.BigInteger(), nullable=False),
        sa.Column('user_answer', sa.Text(), nullable=False),
        sa.Column('is_correct', sa.Boolean(), nullable=False),
        sa.Column('score', sa.Float(), nullable=True),
        sa.Column('kp_id', sa.String(64), nullable=True),
        sa.Column('submitted_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['quiz_item_id'], ['quiz_item.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['user_id'], ['user.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_quiz_attempt_quiz_item_id', 'quiz_attempt', ['quiz_item_id'])
    op.create_index('ix_quiz_attempt_user_id', 'quiz_attempt', ['user_id'])
    op.create_index('ix_quiz_attempt_user_time', 'quiz_attempt', ['user_id', 'submitted_at'])

    # 16. learning_path_item（FK → learning_path + kg_node）
    op.create_table('learning_path_item',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('path_id', sa.BigInteger(), nullable=False),
        sa.Column('kp_id', sa.String(64), nullable=False),
        sa.Column('order_index', sa.Integer(), nullable=False),
        sa.Column('is_completed', sa.Boolean(), nullable=False, server_default='0'),
        sa.ForeignKeyConstraint(['path_id'], ['learning_path.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['kp_id'], ['kg_node.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_learning_path_item_path_id', 'learning_path_item', ['path_id'])

    # 17. learning_record（FK → user + resource_meta + kg_node）
    op.create_table('learning_record',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('user_id', sa.BigInteger(), nullable=False),
        sa.Column('resource_id', sa.BigInteger(), nullable=True),
        sa.Column('kp_id', sa.String(64), nullable=True),
        sa.Column('action', sa.String(64), nullable=False),
        sa.Column('duration_seconds', sa.Integer(), nullable=True),
        sa.Column('recorded_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['user.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['resource_id'], ['resource_meta.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['kp_id'], ['kg_node.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_learning_record_user_id', 'learning_record', ['user_id'])
    op.create_index('ix_learning_record_kp_id', 'learning_record', ['kp_id'])
    op.create_index('ix_learning_record_user_kp', 'learning_record', ['user_id', 'kp_id'])


def _create_tables_uuid() -> None:
    """用 UUID 类型重建所有表（downgrade 用）。"""
    op.create_table('user',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('username', sa.String(64), nullable=False),
        sa.Column('hashed_password', sa.String(128), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('username'),
    )

    op.create_table('document_chunk',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('chunk_id', sa.String(128), nullable=False),
        sa.Column('doc_id', sa.String(128), nullable=False),
        sa.Column('collection_name', sa.String(64), nullable=False, server_default='knowledge_base'),
        sa.Column('text', sa.Text(), nullable=False),
        sa.Column('embedding', sa.JSON(), nullable=True),
        sa.Column('source', sa.String(512), nullable=True),
        sa.Column('page', sa.Integer(), nullable=True),
        sa.Column('section', sa.String(256), nullable=True),
        sa.Column('user_id', sa.String(64), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('chunk_id'),
    )
    op.create_index('ix_document_chunk_chunk_id', 'document_chunk', ['chunk_id'])
    op.create_index('ix_document_chunk_doc_id', 'document_chunk', ['doc_id'])
    op.create_index('ix_document_chunk_collection_name', 'document_chunk', ['collection_name'])

    op.create_table('kg_node',
        sa.Column('id', sa.String(64), nullable=False),
        sa.Column('name', sa.String(256), nullable=False),
        sa.Column('node_type', sa.String(32), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('course_id', sa.String(64), nullable=True),
        sa.Column('user_id', sa.Uuid(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['user.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id'),
    )

    op.create_table('chat_session',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('user_id', sa.Uuid(), nullable=False),
        sa.Column('title', sa.String(256), nullable=True),
        sa.Column('last_used_at', sa.DateTime(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['user.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )

    op.create_table('student_profile',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('user_id', sa.Uuid(), nullable=False),
        sa.Column('major', sa.String(128), nullable=True),
        sa.Column('learning_goal', sa.Text(), nullable=True),
        sa.Column('cognitive_style', sa.String(32), nullable=True),
        sa.Column('daily_time_minutes', sa.Integer(), nullable=True),
        sa.Column('knowledge_mastered', sa.JSON(), nullable=True),
        sa.Column('knowledge_weak', sa.JSON(), nullable=True),
        sa.Column('error_prone', sa.JSON(), nullable=True),
        sa.Column('current_progress', sa.Text(), nullable=True),
        sa.Column('goal_questions', sa.JSON(), nullable=True),
        sa.Column('version', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['user.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('user_id'),
    )

    op.create_table('resource_meta',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('user_id', sa.Uuid(), nullable=False),
        sa.Column('kp_id', sa.String(256), nullable=False),
        sa.Column('resource_type', sa.String(32), nullable=False),
        sa.Column('title', sa.String(256), nullable=True),
        sa.Column('content', sa.Text(), nullable=True),
        sa.Column('content_json', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['user.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )

    op.create_table('learning_path',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('user_id', sa.Uuid(), nullable=False),
        sa.Column('title', sa.String(256), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['user.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )

    op.create_table('kg_build_task',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('doc_id', sa.String(128), nullable=False),
        sa.Column('user_id', sa.Uuid(), nullable=True),
        sa.Column('status', sa.String(32), nullable=False, server_default='pending'),
        sa.Column('progress', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('stage', sa.String(64), nullable=True),
        sa.Column('nodes_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('edges_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['user.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id'),
    )

    op.create_table('generation_batch',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('user_id', sa.Uuid(), nullable=False),
        sa.Column('kp_id', sa.String(256), nullable=False),
        sa.Column('status', sa.String(32), nullable=False, server_default='pending'),
        sa.Column('progress', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('resource_types', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['user.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )

    op.create_table('chat_message',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('session_id', sa.Uuid(), nullable=False),
        sa.Column('role', sa.String(16), nullable=False),
        sa.Column('content', sa.Text(), nullable=True),
        sa.Column('resource_type', sa.String(16), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['session_id'], ['chat_session.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_chat_message_session_time', 'chat_message', ['session_id', 'created_at'])
    op.create_index('ix_chat_message_session_id', 'chat_message', ['session_id'])

    op.create_table('profile_history',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('profile_id', sa.Uuid(), nullable=False),
        sa.Column('snapshot', sa.JSON(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['profile_id'], ['student_profile.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )

    op.create_table('kg_edge',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('source_id', sa.String(64), nullable=False),
        sa.Column('target_id', sa.String(64), nullable=False),
        sa.Column('relation', sa.String(32), nullable=False),
        sa.ForeignKeyConstraint(['source_id'], ['kg_node.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['target_id'], ['kg_node.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('source_id', 'target_id', 'relation'),
    )

    op.create_table('generation_task',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('resource_id', sa.Uuid(), nullable=False),
        sa.Column('batch_id', sa.Uuid(), nullable=True),
        sa.Column('status', sa.String(32), nullable=False, server_default='pending'),
        sa.Column('progress', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['resource_id'], ['resource_meta.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['batch_id'], ['generation_batch.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('resource_id'),
    )

    op.create_table('quiz_item',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('resource_id', sa.Uuid(), nullable=False),
        sa.Column('kp_id', sa.String(64), nullable=True),
        sa.Column('question_type', sa.String(16), nullable=False),
        sa.Column('stem', sa.Text(), nullable=False),
        sa.Column('options', sa.JSON(), nullable=True),
        sa.Column('answer', sa.Text(), nullable=False),
        sa.Column('explanation', sa.Text(), nullable=True),
        sa.Column('order_index', sa.Integer(), nullable=False, server_default='0'),
        sa.ForeignKeyConstraint(['resource_id'], ['resource_meta.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )

    op.create_table('quiz_attempt',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('quiz_item_id', sa.Uuid(), nullable=False),
        sa.Column('user_id', sa.Uuid(), nullable=False),
        sa.Column('user_answer', sa.Text(), nullable=False),
        sa.Column('is_correct', sa.Boolean(), nullable=False),
        sa.Column('score', sa.Float(), nullable=True),
        sa.Column('kp_id', sa.String(64), nullable=True),
        sa.Column('submitted_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['quiz_item_id'], ['quiz_item.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['user_id'], ['user.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )

    op.create_table('learning_path_item',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('path_id', sa.Uuid(), nullable=False),
        sa.Column('kp_id', sa.String(64), nullable=False),
        sa.Column('order_index', sa.Integer(), nullable=False),
        sa.Column('is_completed', sa.Boolean(), nullable=False, server_default='0'),
        sa.ForeignKeyConstraint(['path_id'], ['learning_path.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['kp_id'], ['kg_node.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )

    op.create_table('learning_record',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('user_id', sa.Uuid(), nullable=False),
        sa.Column('resource_id', sa.Uuid(), nullable=True),
        sa.Column('kp_id', sa.String(64), nullable=True),
        sa.Column('action', sa.String(64), nullable=False),
        sa.Column('duration_seconds', sa.Integer(), nullable=True),
        sa.Column('recorded_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['user.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['resource_id'], ['resource_meta.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['kp_id'], ['kg_node.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id'),
    )
