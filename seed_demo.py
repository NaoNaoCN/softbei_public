"""
seed_demo.py
演示数据种子脚本。向数据库写入一套可复现的「演示账号 + 学生画像 + 知识图谱 + 学习路径」，并可直接在这些知识点上生成资源。

用法：
    python seed_demo.py            # 使用默认账号 demo / demo1234
    python seed_demo.py --username demo --password demo1234

运行前提：数据库已 alembic upgrade head，且 .env 中 DATABASE_URL 可连通。
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timedelta

from loguru import logger

from backend.auth.hash_utils import hash_password
from backend.db.database import init_db, close_db, get_session_ctx
from backend.db.crud import insert, insert_many, select_one, delete_user_cascade
from backend.db.models import (
    User,
    StudentProfile,
    KGNode,
    KGEdge,
    LearningPath,
    LearningPathItem,
)

# ----------------------------------------------------------
# 演示知识图谱：以《动手学深度学习》为蓝本的最小课程结构
# 节点 id 全局唯一，course_id 关联到课程根节点
# ----------------------------------------------------------

COURSE_ID = "course_dl"

# (id, name, node_type, description)
KG_NODES = [
    (COURSE_ID, "深度学习", "Course", "基于《动手学深度学习》的深度学习入门课程"),
    ("ch_01", "预备知识与线性神经网络", "Chapter", "张量操作、自动求导、线性回归与 softmax 回归"),
    ("ch_02", "多层感知机", "Chapter", "MLP、激活函数、过拟合与正则化"),
    ("ch_03", "卷积神经网络", "Chapter", "卷积、池化、经典 CNN 架构"),
    # 章节 1 下的知识点
    ("kp_linreg", "线性回归", "KnowledgePoint", "线性模型、损失函数与小批量随机梯度下降"),
    ("kp_softmax", "softmax 回归", "KnowledgePoint", "分类问题、交叉熵损失与 softmax 运算"),
    # 章节 2 下的知识点
    ("kp_mlp", "多层感知机", "KnowledgePoint", "隐藏层、激活函数与前向传播"),
    ("kp_overfit", "过拟合与正则化", "KnowledgePoint", "权重衰减、暂退法（Dropout）"),
    # 章节 3 下的知识点
    ("kp_conv", "卷积运算", "KnowledgePoint", "二维互相关、卷积核、特征映射"),
    ("kp_pool", "池化层", "KnowledgePoint", "最大池化、平均池化与降采样"),
]

# (source_id, target_id, relation)
# IS_PART_OF：章节属于课程 / 知识点属于章节
# REQUIRES：知识点前置依赖
KG_EDGES = [
    ("ch_01", COURSE_ID, "IS_PART_OF"),
    ("ch_02", COURSE_ID, "IS_PART_OF"),
    ("ch_03", COURSE_ID, "IS_PART_OF"),
    ("kp_linreg", "ch_01", "IS_PART_OF"),
    ("kp_softmax", "ch_01", "IS_PART_OF"),
    ("kp_mlp", "ch_02", "IS_PART_OF"),
    ("kp_overfit", "ch_02", "IS_PART_OF"),
    ("kp_conv", "ch_03", "IS_PART_OF"),
    ("kp_pool", "ch_03", "IS_PART_OF"),
    # 前置依赖链
    ("kp_softmax", "kp_linreg", "REQUIRES"),
    ("kp_mlp", "kp_softmax", "REQUIRES"),
    ("kp_overfit", "kp_mlp", "REQUIRES"),
    ("kp_conv", "kp_mlp", "REQUIRES"),
    ("kp_pool", "kp_conv", "REQUIRES"),
]

# 学习路径覆盖的知识点顺序（按依赖拓扑）
LEARNING_PATH_KPS = ["kp_linreg", "kp_softmax", "kp_mlp", "kp_overfit", "kp_conv", "kp_pool"]


async def seed(username: str, password: str, email: str | None) -> None:
    await init_db()
    try:
        async with get_session_ctx() as session:
            # ---- 幂等：先清除同名旧账号的全部关联数据 ----
            existing = await select_one(session, User, filters={"username": username})
            if existing:
                logger.info(f"[seed] 发现同名账号 {username}(id={existing.id})，先级联清除旧数据...")
                await delete_user_cascade(session, existing.id)

            # ---- 1. 账号（bcrypt 加密密码） ----
            user_data = {"username": username, "hashed_password": hash_password(password)}
            if email:
                user_data["email"] = email
                user_data["email_verified"] = True
                user_data["email_verified_at"] = datetime.utcnow()
            user = await insert(session, User, data=user_data)
            logger.info(f"[seed] 创建账号 {username}(id={user.id})")

            # ---- 2. 学生画像 ----
            await insert(session, StudentProfile, data={
                "user_id": user.id,
                "major": "计算机科学与技术",
                "learning_goal": "系统掌握深度学习基础，能够独立实现并训练线性回归、多层感知机与卷积神经网络模型。",
                "cognitive_style": "视觉型",
                "daily_time_minutes": 90,
                "knowledge_mastered": ["线性回归", "softmax 回归"],
                "knowledge_weak": ["过拟合与正则化", "卷积运算"],
                "error_prone": ["交叉熵损失推导", "卷积输出尺寸计算"],
                "current_progress": "已完成前两章，正在学习卷积神经网络。",
                "goal_questions": [
                    "我想入门深度学习，应该从哪里开始？",
                    "怎么理解卷积神经网络？",
                ],
            })
            logger.info("[seed] 创建学生画像")

            # ---- 3. 知识图谱节点 + 边（挂在该用户名下） ----
            await insert_many(session, KGNode, [
                {
                    "id": nid, "name": name, "node_type": ntype,
                    "description": desc, "course_id": COURSE_ID, "user_id": user.id,
                }
                for (nid, name, ntype, desc) in KG_NODES
            ])
            await insert_many(session, KGEdge, [
                {"source_id": s, "target_id": t, "relation": r}
                for (s, t, r) in KG_EDGES
            ])
            logger.info(f"[seed] 创建知识图谱：{len(KG_NODES)} 节点 / {len(KG_EDGES)} 边")

            # ---- 4. 学习路径 ----
            name_map = {nid: name for (nid, name, _t, _d) in KG_NODES}
            path = await insert(session, LearningPath, data={
                "user_id": user.id,
                "title": "深度学习入门路径",
                "description": "从线性回归到卷积神经网络的循序渐进学习路径。",
            })
            await insert_many(session, LearningPathItem, [
                {
                    "path_id": path.id,
                    "kp_id": kp_id,
                    "order_index": i,
                    # 前两个知识点标记为已完成
                    "is_completed": i < 2,
                }
                for i, kp_id in enumerate(LEARNING_PATH_KPS)
            ])
            logger.info(f"[seed] 创建学习路径「{path.title}」，含 {len(LEARNING_PATH_KPS)} 个知识点")

        logger.success(
            f"\n演示数据写入完成。\n"
            f"  账号: {username}\n  密码: {password}\n"
            f"  下一步：登录后在「资源生成」中针对上述知识点生成若干资源、做一次测验，\n"
            f"  即可让演示数据包含真实生成的正文与答题记录，再执行 pg_dump 导出。"
        )
    finally:
        await close_db()


def main() -> None:
    parser = argparse.ArgumentParser(description="写入演示账号与知识图谱种子数据")
    parser.add_argument("--username", default="demo", help="演示账号用户名（默认 demo）")
    parser.add_argument("--password", default="demo1234", help="演示账号密码（默认 demo1234）")
    parser.add_argument("--email", default=None, help="可选邮箱（给定则标记为已验证）")
    args = parser.parse_args()
    asyncio.run(seed(args.username, args.password, args.email))


if __name__ == "__main__":
    main()

