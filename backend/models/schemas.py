"""Pydantic v2 数据模型，供 FastAPI 路由、Agent 以及前端 API 调用共同使用。"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


# 枚举定义

class ResourceType(str, Enum):
    doc = "doc"
    mindmap = "mindmap"
    quiz = "quiz"
    code = "code"
    summary = "summary"
    animation = "animation"
    kg = "kg"


class TaskStatus(str, Enum):
    pending = "pending"
    running = "running"
    done = "done"
    failed = "failed"


class KGNodeType(str, Enum):
    Course = "Course"
    Chapter = "Chapter"
    KnowledgePoint = "KnowledgePoint"
    SubPoint = "SubPoint"
    Concept = "Concept"


class KGRelation(str, Enum):
    IS_PART_OF = "IS_PART_OF"
    REQUIRES = "REQUIRES"
    RELATED_TO = "RELATED_TO"
    CONTAINS = "CONTAINS"


class QuestionType(str, Enum):
    single = "single"
    multi = "multi"
    fill = "fill"
    short = "short"


class CognitiveStyle(str, Enum):
    visual = "视觉型"
    text = "阅读型"
    practice = "动手型"


# 用户相关

# 入站数据（用户提交），包含 password，不含 id/created_at（这些还不存在）
class UserCreate(BaseModel):
    username: str = Field(..., min_length=2, max_length=64)
    password: str = Field(..., min_length=6)
    email: str | None = Field(default=None, max_length=256)

# 出站数据（API 响应），包含 id/created_at，故意没有 password，且有 from_attributes=True 用于从 ORM 对象直接构造
class UserOut(BaseModel):
    id: int
    username: str
    email: str | None = None
    email_verified: bool = False
    created_at: datetime

    model_config = {"from_attributes": True}


class UserUpdate(BaseModel):
    username: str | None = Field(default=None, min_length=2, max_length=64)
    email: str | None = Field(default=None, max_length=256)


class AccountDeleteIn(BaseModel):
    """注销账号请求体：要求同时提供用户名与密码进行双重确认。"""
    username: str = Field(..., min_length=2, max_length=64)
    password: str = Field(..., min_length=6)


class TokenOut(BaseModel):
    """登录成功后返回给前端的响应体，前端拿到 access_token 后在后续请求的 HTTP Header 中携带。"""
    user_id: int
    access_token: str
    token_type: str = "bearer"


# 邮箱验证 & 密码重置

class ForgotPasswordIn(BaseModel):
    email: str = Field(..., max_length=256)


class ResetPasswordIn(BaseModel):
    token: str
    new_password: str = Field(..., min_length=6)


class EmailVerificationOut(BaseModel):
    message: str


# 学生画像

class StudentProfileIn(BaseModel):
    """用户提交 / 更新画像时的请求体"""
    major: Optional[str] = None
    learning_goal: Optional[str] = None
    cognitive_style: Optional[CognitiveStyle] = None
    daily_time_minutes: Optional[int] = Field(None, ge=10, le=480)
    knowledge_mastered: list[str] = Field(default_factory=list)
    knowledge_weak: list[str] = Field(default_factory=list)
    error_prone: list[str] = Field(default_factory=list)
    current_progress: Optional[str] = None
    # 学习目标历史提问：记录每轮对话的用户提问，用于增量概括 learning_goal
    goal_questions: list[str] = Field(default_factory=list)


class StudentProfileOut(StudentProfileIn):
    id: int
    user_id: int
    version: int
    updated_at: datetime

    model_config = {"from_attributes": True}


# 对话会话

class ChatMessageIn(BaseModel):
    content: str = Field(..., min_length=1, max_length=4096)


class ChatMessageOut(BaseModel):
    role: str
    content: str
    resource_type: Optional[str] = None
    extra: Optional[dict[str, Any]] = None
    created_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class ChatSessionOut(BaseModel):
    id: int
    title: Optional[str]
    last_used_at: Optional[datetime] = None
    created_at: datetime

    model_config = {"from_attributes": True}


# 知识图谱

class KGNodeOut(BaseModel):
    id: str
    type: KGNodeType
    name: str
    difficulty: Optional[int]
    is_core: bool
    extra: dict[str, Any] = {}

    model_config = {"from_attributes": True}


class KGEdgeOut(BaseModel):
    source_id: str
    target_id: str
    relation: KGRelation

    model_config = {"from_attributes": True}


class KGGraphOut(BaseModel):
    """完整子图，用于前端 ECharts 渲染"""
    nodes: list[KGNodeOut]
    edges: list[KGEdgeOut]


# 学习路径

class LearningPathItemOut(BaseModel):
    id: int
    order_index: int
    kp_id: str
    kp_name: str
    is_completed: bool

    model_config = {"from_attributes": True}


class LearningPathItemCreate(BaseModel):
    kp_id: str
    order_index: int


class LearningPathItemUpdate(BaseModel):
    order_index: Optional[int] = None
    is_completed: Optional[bool] = None


class LearningPathOut(BaseModel):
    id: int
    name: str
    description: Optional[str]
    items: list[LearningPathItemOut] = []
    created_at: datetime

    model_config = {"from_attributes": True}


class LearningPathCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=256)
    description: Optional[str] = None


class LearningPathUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=256)
    description: Optional[str] = None


# 资源生成

class GenerateRequest(BaseModel):
    """触发资源生成的请求体"""
    kp_id: str = Field(..., description="目标知识点节点 ID")
    resource_type: ResourceType
    num_questions: int = Field(default=4, ge=1, le=20, description="测验题目数量（仅 quiz 类型生效）")
    question_type_counts: dict[str, int] = Field(
        default_factory=dict,
        description="测验各题型数量，如 {\"single\": 5, \"multi\": 3, \"fill\": 2}，合计不超过 num_questions"
    )
    extra_params: dict[str, Any] = Field(default_factory=dict)


class GenerateTaskOut(BaseModel):
    task_id: int
    status: TaskStatus
    progress: int = Field(ge=0, le=100)
    error_message: Optional[str] = None
    result_id: Optional[int] = None

    model_config = {"from_attributes": True}


class KGBuildTaskOut(BaseModel):
    task_id: int
    doc_id: str
    status: TaskStatus
    progress: int = Field(ge=0, le=100)
    stage: Optional[str] = None
    nodes_count: int = 0
    edges_count: int = 0
    error_message: Optional[str] = None

    model_config = {"from_attributes": True}


class BatchGenerateRequest(BaseModel):
    """批量资源生成请求"""
    kp_id: str = Field(..., description="目标知识点节点 ID")
    resource_types: list[ResourceType] = Field(..., min_length=1, description="要生成的资源类型列表")
    num_questions: int = Field(default=4, ge=1, le=20, description="测验题目数量（仅 quiz 类型生效）")
    question_type_counts: dict[str, int] = Field(
        default_factory=dict,
        description="测验各题型数量，如 {\"single\": 5, \"multi\": 3, \"fill\": 2}"
    )


class BatchTaskItem(BaseModel):
    """批次中单个子任务的状态"""
    task_id: int
    resource_type: ResourceType
    status: TaskStatus
    progress: int = Field(ge=0, le=100)
    result_id: Optional[int] = None
    error_message: Optional[str] = None


class BatchGenerateOut(BaseModel):
    """批量生成响应"""
    batch_id: int
    status: TaskStatus
    progress: int = Field(ge=0, le=100)
    tasks: list[BatchTaskItem] = Field(default_factory=list)


class ResourceMetaOut(BaseModel):
    id: int
    user_id: int
    kp_id: Optional[str]
    resource_type: ResourceType
    title: str
    content: Optional[str]
    content_json: Optional[dict[str, Any]]
    created_at: datetime

    model_config = {"from_attributes": True}


class ResourceListOut(BaseModel):
    items: list[ResourceMetaOut]
    total: int


# 测验 / 题目

class QuizItemOut(BaseModel):
    id: int
    kp_id: Optional[str]
    question_type: QuestionType
    difficulty: Optional[int]
    stem: str
    options: Optional[list[str]]
    answer: Any
    explanation: Optional[str]

    model_config = {"from_attributes": True}


class QuizSubmitIn(BaseModel):
    """学生提交答题结果"""
    quiz_item_id: int
    user_answer: Any
    duration_seconds: Optional[int] = None  # 可选：前端上报做题用时（秒）


class QuizAttemptOut(BaseModel):
    id: int
    quiz_item_id: int
    user_answer: Any
    is_correct: bool
    score: float
    kp_id: Optional[str] = None
    kp_name: Optional[str] = None
    created_at: datetime
    # 题目详情（来自 QuizItem）
    stem: Optional[str] = None
    options: Optional[list[str]] = None
    answer: Optional[Any] = None
    explanation: Optional[str] = None
    question_type: Optional[str] = None
    difficulty: Optional[int] = None

    model_config = {"from_attributes": True}


# 学习记录

class LearningRecordCreate(BaseModel):
    resource_id: Optional[int] = None
    kp_id: Optional[str] = None
    action: str = "view"          # "view" | "quiz" | "complete"
    duration_seconds: Optional[int] = None


class LearningRecordOut(BaseModel):
    id: int
    user_id: int
    resource_id: Optional[int] = None
    kp_id: Optional[str] = None
    kp_name: Optional[str] = None
    action: str
    duration_seconds: Optional[int] = None
    recorded_at: datetime

    model_config = {"from_attributes": True}


# 学习计划表（Study Plan）

class StudyPlanResourceRef(BaseModel):
    """计划项关联的已有资源的轻量引用（不含正文内容）。"""
    resource_id: int
    resource_type: ResourceType
    title: str

    model_config = {"from_attributes": True}


class StudyPlanItemOut(BaseModel):
    id: int
    kp_id: Optional[str] = None
    kp_name: str
    scheduled_date: date
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    estimated_minutes: Optional[int] = None
    order_index: int
    is_completed: bool
    resources: list[StudyPlanResourceRef] = Field(default_factory=list)
    missing_resource_types: list[str] = Field(default_factory=list)
    notes: Optional[str] = None

    model_config = {"from_attributes": True}


class StudyPlanOut(BaseModel):
    id: int
    user_id: int
    title: Optional[str] = None
    description: Optional[str] = None
    goal: Optional[str] = None
    start_date: date
    end_date: date
    daily_time_minutes: Optional[int] = None
    status: str
    source_path_ids: list[int] = Field(default_factory=list)
    items: list[StudyPlanItemOut] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class StudyPlanGenerateRequest(BaseModel):
    """触发学习计划生成的请求体。"""
    source: str = Field(default="aggregate", description="aggregate=汇总已有路径+画像补全；path=指定单条路径")
    path_ids: list[int] = Field(default_factory=list, description="source=path 时指定的 LearningPath id 列表")
    start_date: Optional[date] = Field(default=None, description="计划起始日期，默认今天")
    days: Optional[int] = Field(default=None, ge=1, le=180, description="目标完成天数，给定则均摊每日预算")
    daily_time_minutes: Optional[int] = Field(default=None, ge=10, le=720, description="覆盖画像中的每日学习预算")
    default_start_hour: Optional[str] = Field(default=None, description="每日起始时刻，如 '19:00'；给定则生成时段")
    title: Optional[str] = Field(default=None, max_length=256, description="自定义计划标题")


class StudyPlanItemUpdate(BaseModel):
    scheduled_date: Optional[date] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    order_index: Optional[int] = None
    is_completed: Optional[bool] = None
    notes: Optional[str] = None


class StudyPlanUpdate(BaseModel):
    title: Optional[str] = Field(default=None, max_length=256)
    description: Optional[str] = None
    status: Optional[str] = None


# Agent 内部状态（LangGraph State）

class AgentState(BaseModel):
    """LangGraph 全局状态，在各 Agent 节点间传递"""
    user_id: int
    session_id: int
    user_message: str
    chat_history: list[dict[str, str]] = Field(default_factory=list)
    intent_type: Optional[str] = None  # "generate" | "clarify"
    profile: Optional[StudentProfileIn] = None
    kp_id: Optional[str] = None
    resource_type: Optional[ResourceType] = None
    retrieved_docs: list[str] = Field(default_factory=list)
    draft_content: Optional[str] = None
    final_content: Optional[str] = None
    safety_passed: bool = True
    error: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    kg_doc_id: Optional[str] = None       # 知识图谱构建的目标文档 ID
    # 画像初始化 / 追问流程控制
    is_onboarding: bool = False          # 前端标记：当前是否处于画像初始化阶段
    profile_complete: bool = False       # profile_agent 判断后写入，供条件路由使用
    clarify_message: Optional[str] = None  # 追问内容，情况A/B时写入，透传给前端
    num_questions: int = 4
    question_type_counts: dict[str, int] = Field(default_factory=dict)  # 各题型数量，如 {"single":5,"multi":3,"fill":2}