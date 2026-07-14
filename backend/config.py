"""配置定义 — Pydantic BaseModel 自动映射，从 configs/config.yaml 加载。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

# 加载项目根目录的 .env 文件（os.getenv() 优先于 .env）
load_dotenv(Path(__file__).parent.parent / ".env", override=False)


class LLMRetryConfig(BaseModel):
    max_attempts: int = 5
    backoff_multiplier: int = 2
    backoff_min_seconds: int = 3
    backoff_max_seconds: int = 30


class LLMTimeoutConfig(BaseModel):
    connect: int = 10
    read: int = 120
    write: int = 30
    pool: int = 10


class LLMProviderConfig(BaseModel):
    """单个 LLM Provider 配置"""
    base_url: str = ""
    default_model: str = ""
    api_key: str = ""      # 该 provider 专用 key；留空则回退到全局 llm.api_key


class LLMProvidersConfig(BaseModel):
    """所有 LLM Provider 配置"""
    spark: LLMProviderConfig = Field(default_factory=LLMProviderConfig)
    deepseek: LLMProviderConfig = Field(default_factory=LLMProviderConfig)
    qwen: LLMProviderConfig = Field(default_factory=LLMProviderConfig)
    openai: LLMProviderConfig = Field(default_factory=LLMProviderConfig)


class LLMConfig(BaseModel):
    api_key: str = ""
    base_url: str = ""
    model: str = ""
    provider: str = "qwen"
    provider_order: list[str] = Field(default_factory=list)  # 级联 failover 顺序；空=只用 provider，不切换
    default_max_tokens: int = 2048
    enable_thinking: bool | None = None  # Qwen3 深度思考开关；None=不传该参数（兼容非 Qwen3 模型）
    retry: LLMRetryConfig = Field(default_factory=LLMRetryConfig)
    timeout: LLMTimeoutConfig = Field(default_factory=LLMTimeoutConfig)
    providers: LLMProvidersConfig = Field(default_factory=LLMProvidersConfig)


class DatabaseConfig(BaseModel):
    url: str = ""
    echo: bool = False
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600
    command_timeout: int = 60


class VectorDBConfig(BaseModel):
    collection: str = "knowledge_base"
    hnsw_ef_search: int = 100          # HNSW 检索精度，越大越精确但越慢


class EmbeddingConfig(BaseModel):
    use_spark: bool = True
    concurrency: int = 8
    api_model: str = "text-embedding-v4"
    api_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key: str = ""      # Embedding 专用 key；留空则回退到全局 llm.api_key
    timeout_read: int = 60
    timeout_connect: int = 10
    timeout_write: int = 30
    timeout_pool: int = 10
    index_batch_size: int = 128
    api_max_batch_size: int = 10       # Embedding API 单次调用最大条数（provider 限制）
    vector_dimension: int = 1024


class ParentChunkingConfig(BaseModel):
    enabled: bool = False
    parent_max_chars: int = 2000
    child_chunk_size: int | None = None   # None = 使用 rag.chunk_size
    child_chunk_overlap: int = 100
    score_weight: str = "max"             # "max"（取子块最高分）| "mean"（子块均分）
    parent_split_lookback: int = 400      # 父块切分时向前搜索安全边界的最大字符数


class HybridConfig(BaseModel):
    """多路召回 — 混合检索配置"""
    enabled: bool = False
    paths: list[str] = Field(default_factory=lambda: ["vector", "keyword"])
    rrf_k: int = 60
    vector_weight: float = 1.0
    keyword_weight: float = 1.0


class RAGConfig(BaseModel):
    chunk_size: int = 500
    chunk_overlap: int = 50
    n_results: int = 5
    score_threshold: float = 0.5
    context_max_tokens: int = 3000
    max_sections_before_coarse_split: int = 50
    parent_chunking: ParentChunkingConfig = Field(default_factory=ParentChunkingConfig)
    hybrid: HybridConfig = Field(default_factory=HybridConfig)
    # -- 检索精排 --
    re_rank_keyword_boost: float = 0.25   # jieba 关键词重叠加权上限
    keyword_min_length: int = 2           # jieba 分词后有效关键词的最小长度（字符）
    keyword_score_threshold: float = 0.0  # 关键词检索最低分数阈值（0=不设限）
    prefetch_multiplier: int = 3          # 预取倍数（n_results × prefetch_multiplier 候选供精排）
    prefetch_min: int = 15                # 预取候选数下限
    # -- 切分精调 --
    split_sentence_lookback_ratio: float = 0.2  # 句边界搜索回看比例（占 chunk_size）
    min_heading_level_for_split: int = 3        # 自适应粗切分的最细标题级别（≤2=H1/H2 才允许升粗）
    # -- Query Rewrite 子配置 --
    query_rewrite_enabled: bool = True
    query_rewrite_decontextualize: bool = True
    query_rewrite_profile_aware: bool = True
    query_rewrite_multi_query: bool = False
    query_rewrite_multi_query_count: int = 3
    query_rewrite_temperature: float = 0.1
    query_rewrite_max_tokens: int = 150


class TokenEstimationConfig(BaseModel):
    cn_chars_per_token: float = 1.5
    en_chars_per_token: float = 4.0


class ChatConfig(BaseModel):
    max_turns: int = 10
    history_max_tokens: int = 4000
    message_max_length: int = 4096
    session_expiry_days: int = 30
    cleanup_interval_hours: int = 24
    auto_title_max_chars: int = 15
    auto_title_message_truncate: int = 200
    auto_title_max_tokens: int = 30
    auto_title_final_length: int = 20
    token_estimation: TokenEstimationConfig = Field(default_factory=TokenEstimationConfig)


class KnowledgeGraphConfig(BaseModel):
    llm_concurrency: int = 10
    max_batches: int = 30
    toc_max_items: int = 100
    batch_chars_limit: int = 12000
    text_truncate_chars: int = 6000
    node_extraction_max_tokens: int = 4000
    edge_batch_size: int = 40
    edge_overlap: int = 10
    section_merge_min_chars: int = 200


class GenerationQuizConfig(BaseModel):
    """题目生成 — 弱项阈值与配比"""
    weak_threshold_high: int = 5
    weak_threshold_mid: int = 2
    counts_high: list[int] = Field(default_factory=lambda: [3, 2, 2])
    counts_mid: list[int] = Field(default_factory=lambda: [2, 1, 1])
    counts_default: list[int] = Field(default_factory=lambda: [2, 1, 1])


class GenerationConfig(BaseModel):
    default_num_questions: int = 4
    max_questions: int = 20
    mindmap_max_depth: int = 4
    mindmap_max_children: int = 6
    quiz: GenerationQuizConfig = Field(default_factory=GenerationQuizConfig)


class StorageCleanupConfig(BaseModel):
    enabled: bool = True
    retention_days: int = 30
    orphan_retention_days: int = 7
    interval_hours: int = 24
    min_file_age_seconds: int = 300


class StorageConfig(BaseModel):
    upload_dir: str = "uploaded_docs"
    knowledge_base_dir: str = "knowledge_base"
    supported_extensions: list[str] = Field(default_factory=lambda: [".pdf", ".docx", ".doc", ".md", ".txt"])
    doc_id_hex_length: int = 12
    upload_enabled: bool = True  # 为 false 时禁止上传并索引文档（其他功能不受影响）
    upload_disabled_message: str = "文档上传与索引功能暂时关闭。"  # 上传被禁用时返回给前端的提示文案
    cleanup: StorageCleanupConfig = Field(default_factory=StorageCleanupConfig)


class LoggingConfig(BaseModel):
    dir: str = "logs"
    retention_days: int = 30
    error_retention_days: int = 90
    trace_id_length: int = 8
    json_format: bool = False       # JSON 结构化日志（生产环境推荐开启）
    level: str = "DEBUG"            # 文件日志级别
    console_level: str = "INFO"     # 控制台日志级别


class JWTConfig(BaseModel):
    secret: str = ""
    algorithm: str = "HS256"
    expire_hours: int = 24


class EmailConfig(BaseModel):
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_from: str = ""
    smtp_use_tls: bool = True
    smtp_timeout: int = 30
    max_retries: int = 3
    verification_expire_minutes: int = 30
    password_reset_expire_minutes: int = 15
    rate_limit_send_per_hour: int = 5

    @field_validator("smtp_port", mode="before")
    @classmethod
    def _coerce_port(cls, v):
        if v == "" or v is None:
            return 587
        return int(v)


class ClarifyAgentConfig(BaseModel):
    temperature: float = 0.7


class CodeAgentConfig(BaseModel):
    temperature: float = 0.7
    max_tokens: int = 5000


class AnimAgentConfig(BaseModel):
    """Animation Agent 配置（基于 p5.js 的教学动画生成）"""
    temperature: float = 0.4
    max_tokens: int = 6000


class DocAgentConfig(BaseModel):
    temperature: float = 0.7
    max_tokens: int = 4000


class MindmapAgentConfig(BaseModel):
    temperature: float = 0.5
    max_tokens: int = 2000


class PlannerAgentConfig(BaseModel):
    intent_temperature: float = 0.0
    classify_temperature: float = 0.1
    smart_plan_temperature: float = 0.3
    history_lookback_messages: int = 6
    fallback_kp_id_length: int = 50
    smart_plan_default_types: list[str] = Field(default_factory=lambda: ["doc", "quiz"])


class ProfileAgentConfig(BaseModel):
    extract_temperature: float = 0.1
    intent_temperature: float = 0.0
    clarify_temperature: float = 0.7
    goal_summary_temperature: float = 0.3
    max_goal_questions: int = 50
    history_max_versions: int = 10


class QuizAgentConfig(BaseModel):
    temperature: float = 0.6
    max_tokens: int = 3000


class RecommendAgentConfig(BaseModel):
    temperature: float = 0.7
    max_tokens: int = 2000
    min_recommendations: int = 3
    max_recommendations: int = 5


class SafetyAgentConfig(BaseModel):
    temperature: float = 0.1
    max_tokens: int = 300
    max_ref_docs: int = 3
    draft_preview_chars: int = 500


class SummaryAgentConfig(BaseModel):
    temperature: float = 0.7
    max_tokens: int = 1200
    target_words_min: int = 300
    target_words_max: int = 500


class AgentsConfig(BaseModel):
    clarify: ClarifyAgentConfig = Field(default_factory=ClarifyAgentConfig)
    code: CodeAgentConfig = Field(default_factory=CodeAgentConfig)
    anim: AnimAgentConfig = Field(default_factory=AnimAgentConfig)
    doc: DocAgentConfig = Field(default_factory=DocAgentConfig)
    mindmap: MindmapAgentConfig = Field(default_factory=MindmapAgentConfig)
    planner: PlannerAgentConfig = Field(default_factory=PlannerAgentConfig)
    profile: ProfileAgentConfig = Field(default_factory=ProfileAgentConfig)
    quiz: QuizAgentConfig = Field(default_factory=QuizAgentConfig)
    recommend: RecommendAgentConfig = Field(default_factory=RecommendAgentConfig)
    safety: SafetyAgentConfig = Field(default_factory=SafetyAgentConfig)
    summary: SummaryAgentConfig = Field(default_factory=SummaryAgentConfig)


class VideoSearchConfig(BaseModel):
    enabled: bool = True
    max_results: int = 3
    min_query_length: int = 2
    bilibili_timeout: int = 5
    tavily_api_key: str = ""
    tavily_timeout: int = 8


class PaginationConfig(BaseModel):
    default_limit: int = 20
    quiz_attempts_limit: int = 50


class ServerConfig(BaseModel):
    version: str = "0.1.0"
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])


class AuthConfig(BaseModel):
    bcrypt_rounds: int = 12


class EvaluationSamplingConfig(BaseModel):
    development: float = 1.0       # 开发模式：全量评估
    production: float = 0.1        # 生产模式：10% 采样


class EvaluationCrossValidationConfig(BaseModel):
    """多 LLM 交叉验证配置"""
    enabled: bool = False
    providers: list[str] = Field(default_factory=lambda: ["qwen", "deepseek"])


class EvaluationGoldenDatasetConfig(BaseModel):
    """黄金测试集配置"""
    path: str = "backend/evaluation/golden_queries.yaml"
    auto_run_interval_hours: int = 24


class EvaluationABExperimentConfig(BaseModel):
    enabled: bool = False
    groups: list[str] = Field(default_factory=list)


class EvaluationStorageConfig(BaseModel):
    persist_to_db: bool = True
    retention_days: int = 90


class EvaluationConfig(BaseModel):
    enabled: bool = True            # 总开关：关闭后跳过所有评估采集与 LLM Judge
    mode: str = "development"      # "development" | "production"
    health_check_enabled: bool = True
    sampling: EvaluationSamplingConfig = Field(default_factory=EvaluationSamplingConfig)
    cross_validation: EvaluationCrossValidationConfig = Field(default_factory=EvaluationCrossValidationConfig)
    golden_dataset: EvaluationGoldenDatasetConfig = Field(default_factory=EvaluationGoldenDatasetConfig)
    ab_experiment: EvaluationABExperimentConfig = Field(default_factory=EvaluationABExperimentConfig)
    storage: EvaluationStorageConfig = Field(default_factory=EvaluationStorageConfig)


class StudyPlanSequenceConfig(BaseModel):
    """学习计划知识点排序（LLM）配置"""
    temperature: float = 0.3
    max_tokens: int = 2048


class StudyPlanConfig(BaseModel):
    default_daily_minutes: int = 60        # 画像未设置每日时长时的兜底值
    default_start_hour: str = "19:00"      # 默认每日起始时刻（仅在请求要求生成时段时使用）
    default_horizon_days: int = 14         # 未指定 days 时的默认排程跨度上限参考
    min_kp_minutes: int = 20               # 单知识点学习时长下限（钳制 LLM 预估）
    max_kp_minutes: int = 180              # 单知识点学习时长上限
    default_kp_minutes: int = 45           # LLM 未给出预估时的默认单点时长
    # 计划项默认尝试匹配/补全的资源类型
    target_resource_types: list[str] = Field(default_factory=lambda: ["doc", "mindmap", "quiz"])
    sequence: StudyPlanSequenceConfig = Field(default_factory=StudyPlanSequenceConfig)


class Config(BaseModel):
    """全局配置 — 由 model_validate() 从 YAML 自动构建，无需手工映射。"""
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    vector_db: VectorDBConfig = Field(default_factory=VectorDBConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    chat: ChatConfig = Field(default_factory=ChatConfig)
    knowledge_graph: KnowledgeGraphConfig = Field(default_factory=KnowledgeGraphConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    jwt: JWTConfig = Field(default_factory=JWTConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    pagination: PaginationConfig = Field(default_factory=PaginationConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    video_search: VideoSearchConfig = Field(default_factory=VideoSearchConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    study_plan: StudyPlanConfig = Field(default_factory=StudyPlanConfig)
    email: EmailConfig = Field(default_factory=EmailConfig)


def _resolve_env_vars(value: Any) -> Any:
    """递归解析 ${ENV_VAR} 和 ${ENV_VAR-default} 格式的环境变量引用"""
    if isinstance(value, str):
        if value.startswith("${") and value.endswith("}"):
            inner = value[2:-1]
            if "-" in inner:
                env_var, default = inner.split("-", 1)
                return os.getenv(env_var, default)
            return os.getenv(inner, "")
        return value
    elif isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_resolve_env_vars(item) for item in value]
    return value


def _load_config() -> Config:
    """加载 YAML → 解析 ${ENV_VAR} → model_validate 自动映射到 Config。"""
    config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    resolved = _resolve_env_vars(raw)
    return Config.model_validate(resolved)


def _load_prompts() -> dict[str, Any]:
    """Load agent prompts from configs/prompts.yaml."""
    prompts_path = Path(__file__).parent.parent / "configs" / "prompts.yaml"
    if not prompts_path.exists():
        return {}
    with open(prompts_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class PromptsConfig:
    """Agent prompts loaded from prompts.yaml, with config-value resolution."""

    def __init__(self, data: dict, app_config: Config):
        self._data = data
        self._cfg = app_config

    def get(self, path: str) -> str:
        """
        Get a prompt template by dot-separated path.
        Resolves config-level placeholders like {min_recommendations}, {target_words_min}, etc.
        Returns the template with runtime placeholders left intact for later .format().
        """
        val = self._data
        for key in path.split("."):
            if isinstance(val, dict):
                val = val.get(key)
            else:
                return ""
        if not isinstance(val, str):
            return ""

        config_vars = {
            "{min_recommendations}": str(self._cfg.agents.recommend.min_recommendations),
            "{max_recommendations}": str(self._cfg.agents.recommend.max_recommendations),
            "{target_words_min}": str(self._cfg.agents.summary.target_words_min),
            "{target_words_max}": str(self._cfg.agents.summary.target_words_max),
            "{min_kp_minutes}": str(self._cfg.study_plan.min_kp_minutes),
            "{max_kp_minutes}": str(self._cfg.study_plan.max_kp_minutes),
        }
        for key, value in config_vars.items():
            val = val.replace(key, value)
        return val


# 全局单例
config = _load_config()
prompts = PromptsConfig(_load_prompts(), config)

