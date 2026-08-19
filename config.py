"""
全局配置。LLM 供应商相关全部从 .env 读，不写死某个厂商。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    return float(raw) if raw else default

# --------------------------------------------------------------------------
# 路径
# --------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
PROPOSALS_DIR = DATA_DIR / "proposals"
RESEARCH_DIR = DATA_DIR / "research"
ARTICLES_DATA_DIR = DATA_DIR / "articles"
RUNS_DIR = DATA_DIR / "runs"
SOURCE_HEALTH_DIR = DATA_DIR / "source_health"
DOCS_DIR = ROOT_DIR / "docs"
TEMPLATES_DIR = ROOT_DIR / "render" / "templates"

for _dir in (PROPOSALS_DIR, RESEARCH_DIR, ARTICLES_DATA_DIR, RUNS_DIR, SOURCE_HEALTH_DIR, DOCS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# 产品文案
# --------------------------------------------------------------------------
SITE_NAME = "深潜 AI 周刊"
SITE_NAME_EN = "Deep Dive AI Weekly"
SITE_SLOGAN = "每周潜入一个 AI 方向的深处"
SITE_AUTHOR = "AI 编辑部"
SITE_BASE_URL = os.environ.get("SITE_BASE_URL", "")

# --------------------------------------------------------------------------
# LLM（OpenAI 兼容接口，换供应商只改 .env）
# 支持双供应商：默认档（贵/强）+ 便宜档（轻量阶段）
# --------------------------------------------------------------------------
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.moonshot.cn/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "kimi-k3").strip()

# 某些模型（如 kimi-k3）只允许固定 temperature；设置后会覆盖代码里的 temperature 参数
_raw_temp = os.environ.get("LLM_TEMPERATURE", "").strip()
LLM_TEMPERATURE: float | None = float(_raw_temp) if _raw_temp else None


def _parse_extra_kwargs(env_name: str = "LLM_EXTRA_KWARGS") -> dict:
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{env_name} 不是合法 JSON: {raw}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f'{env_name} 必须是 JSON 对象，例如 {{"reasoning_effort":"max"}}')
    return value


LLM_EXTRA_KWARGS = _parse_extra_kwargs("LLM_EXTRA_KWARGS")

# 便宜供应商（可选）。配置后，轻量阶段可走另一套 key/base_url
LLM_CHEAP_API_KEY = os.environ.get("LLM_CHEAP_API_KEY", "").strip()
LLM_CHEAP_BASE_URL = os.environ.get("LLM_CHEAP_BASE_URL", "").strip()
LLM_CHEAP_MODEL = os.environ.get("LLM_CHEAP_MODEL", "").strip()  # 低档，如 deepseek-v4-flash
LLM_MID_MODEL = os.environ.get("LLM_MID_MODEL", "").strip()  # 中档，通常也走便宜供应商
LLM_CHEAP_EXTRA_KWARGS = _parse_extra_kwargs("LLM_CHEAP_EXTRA_KWARGS")
_raw_cheap_temp = os.environ.get("LLM_CHEAP_TEMPERATURE", "").strip()
LLM_CHEAP_TEMPERATURE: float | None = float(_raw_cheap_temp) if _raw_cheap_temp else None


def _model(env_name: str, default: str = "") -> str:
    return os.environ.get(env_name, "").strip() or default or LLM_MODEL


# 各阶段模型；不填则用默认 LLM_MODEL。可用 LLM_CHEAP_MODEL / LLM_MID_MODEL 的值。
MODEL_TOPIC_CLUSTERING = _model("LLM_MODEL_TOPIC_CLUSTERING", LLM_CHEAP_MODEL)
MODEL_TOPIC_SCORING = _model("LLM_MODEL_TOPIC_SCORING", LLM_MID_MODEL or LLM_CHEAP_MODEL)
MODEL_FACTCHECK = _model("LLM_MODEL_FACTCHECK", LLM_CHEAP_MODEL)
MODEL_EDITORIAL = _model("LLM_MODEL_EDITORIAL", LLM_MID_MODEL or LLM_CHEAP_MODEL or LLM_MODEL)
MODEL_MECHANISM = _model("LLM_MODEL_MECHANISM", LLM_MID_MODEL or LLM_MODEL)
MODEL_RESEARCH_AGENT = _model("LLM_MODEL_RESEARCH_AGENT", LLM_MODEL)
MODEL_WRITING = _model("LLM_MODEL_WRITING", LLM_MODEL)
MODEL_WRITING_FALLBACK = _model("LLM_MODEL_WRITING_FALLBACK", MODEL_EDITORIAL)
MODEL_FACTCHECK_FALLBACK = _model("LLM_MODEL_FACTCHECK_FALLBACK", MODEL_WRITING_FALLBACK)


def llm_uses_cheap_provider(model: str) -> bool:
    """模型名落在便宜/中档名单，且已配置便宜供应商时，走便宜档 client。"""
    if not LLM_CHEAP_API_KEY or not LLM_CHEAP_BASE_URL:
        return False
    cheap_models = {m for m in (LLM_CHEAP_MODEL, LLM_MID_MODEL) if m}
    return model in cheap_models

# --------------------------------------------------------------------------
# 其它密钥
# --------------------------------------------------------------------------
SEMANTIC_SCHOLAR_API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
GITHUB_REPOSITORY = os.environ.get("GITHUB_REPOSITORY", "")

# --------------------------------------------------------------------------
# 深度研究 Agent / 事实核查预算
# --------------------------------------------------------------------------
RESEARCH_MAX_TOOL_CALLS = _env_int("RESEARCH_MAX_TOOL_CALLS", 40)
RESEARCH_MAX_RESULTS_PER_SEARCH = _env_int("RESEARCH_MAX_RESULTS_PER_SEARCH", 10)
FACTCHECK_MAX_REVISION_ROUNDS = _env_int("FACTCHECK_MAX_REVISION_ROUNDS", 2)
ARTICLE_MIN_FINAL_SCORE = _env_float("ARTICLE_MIN_FINAL_SCORE", 3.5)
ARTICLE_CLAIM_AUDIT_ROUNDS = _env_int("ARTICLE_CLAIM_AUDIT_ROUNDS", 3)
ARTICLE_REVISION_MAX_ROUNDS = _env_int("ARTICLE_REVISION_MAX_ROUNDS", 3)
ARTICLE_REVISION_MAX_REPAIRS_PER_ROUND = _env_int("ARTICLE_REVISION_MAX_REPAIRS_PER_ROUND", 2)

# 文章结构预算：overview + 机制节 + synthesis
ARTICLE_MAX_SECTIONS = _env_int("ARTICLE_MAX_SECTIONS", 4)
ARTICLE_MIN_BODY_CHARS = _env_int("ARTICLE_MIN_BODY_CHARS", 2200)
ARTICLE_MIN_MECHANISM_RATIO = _env_float("ARTICLE_MIN_MECHANISM_RATIO", 0.45)
ARTICLE_MIN_PRIMARY_MECHANISM_CHARS = _env_int("ARTICLE_MIN_PRIMARY_MECHANISM_CHARS", 650)

# 机制卡助写：≥ min_high 张可用卡才发短深文，否则 insufficient 跳过发布
MECHANISM_CARD_CANDIDATES = _env_int("MECHANISM_CARD_CANDIDATES", 5)
MECHANISM_MIN_HIGH_CARDS = _env_int("MECHANISM_MIN_HIGH_CARDS", 1)
MECHANISM_DEEP_DIVE_CARDS = _env_int("MECHANISM_DEEP_DIVE_CARDS", 2)

# --------------------------------------------------------------------------
# 每周广度扫描信源
# --------------------------------------------------------------------------
# enabled=False 可临时关闭；fetcher 默认 rss，无官方 RSS 的源用 html_list。
RSS_SOURCES = [
    {"name": "OpenAI News", "url": "https://openai.com/news/rss.xml", "authority": 1.0, "lang": "en", "enabled": True},
    # Anthropic 无官方 RSS，改为抓新闻列表页
    {
        "name": "Anthropic News",
        "url": "https://www.anthropic.com/news",
        "authority": 1.0,
        "lang": "en",
        "enabled": True,
        "fetcher": "html_list",
        "path_prefix": "/news/",
    },
    {"name": "Google DeepMind Blog", "url": "https://deepmind.google/blog/rss.xml", "authority": 1.0, "lang": "en", "enabled": True},
    {"name": "Meta AI Blog", "url": "https://ai.meta.com/blog/rss/", "homepage": "https://ai.meta.com/blog/", "authority": 1.0, "lang": "en", "enabled": True, "fallback_fetcher": "html_list", "path_prefix": "/blog/"},
    {"name": "Hugging Face Blog", "url": "https://huggingface.co/blog/feed.xml", "authority": 0.85, "lang": "en", "enabled": True},
    {"name": "Mistral AI News", "url": "https://mistral.ai/news/rss.xml", "homepage": "https://mistral.ai/news/", "authority": 0.9, "lang": "en", "enabled": True, "fallback_fetcher": "html_list", "path_prefix": "/news/"},
    {"name": "TechCrunch AI", "url": "https://techcrunch.com/category/artificial-intelligence/feed/", "authority": 0.6, "lang": "en", "enabled": True},
    {"name": "VentureBeat AI", "url": "https://venturebeat.com/category/ai/feed/", "authority": 0.6, "lang": "en", "enabled": True},
    {"name": "The Verge AI", "url": "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml", "authority": 0.55, "lang": "en", "enabled": True},
    {"name": "Ars Technica AI", "url": "https://arstechnica.com/ai/feed/", "authority": 0.6, "lang": "en", "enabled": True},
    # /rss 已重定向到数据服务页，不再是 feed；有可用地址后再打开
    {"name": "机器之心", "url": "https://www.jiqizhixin.com/rss", "authority": 0.6, "lang": "zh", "enabled": False},
    {"name": "量子位", "url": "https://www.qbitai.com/feed", "authority": 0.6, "lang": "zh", "enabled": True},
]

HN_KEYWORDS = [
    "AI", "LLM", "GPT", "language model", "agent", "diffusion", "transformer",
    "OpenAI", "Anthropic", "DeepMind", "Claude", "Gemini", "Llama", "reinforcement learning",
]
HN_MIN_POINTS = 30

SOURCE_QUARANTINE_AFTER_FAILURES = _env_int("SOURCE_QUARANTINE_AFTER_FAILURES", 3)
SOURCE_QUARANTINE_RETRY_HOURS = _env_int("SOURCE_QUARANTINE_RETRY_HOURS", 24)

ARXIV_CATEGORIES = ["cs.AI", "cs.CL", "cs.LG", "cs.MA"]
ARXIV_MAX_RESULTS_PER_CATEGORY = 60

GITHUB_TRENDING_LANGUAGES = ["python", "jupyter-notebook"]

SCORE_WEIGHTS = {
    "source_diversity": 0.35,
    "source_authority": 0.25,
    "community_buzz": 0.25,
    "novelty": 0.15,
}

# Weekly Deep Dive 选题漏斗。周频约束扫描与研究，不强制每周发布。
TOPIC_SHORTLIST_SIZE = _env_int("TOPIC_SHORTLIST_SIZE", 5)
TOPIC_PROBE_SIZE = _env_int("TOPIC_PROBE_SIZE", 3)
TOPIC_PROBE_DOCS_PER_CANDIDATE = _env_int("TOPIC_PROBE_DOCS_PER_CANDIDATE", 2)
TOPIC_MIN_EDITORIAL_SCORE = _env_float("TOPIC_MIN_EDITORIAL_SCORE", 3.6)
TOPIC_MIN_FEASIBILITY_SCORE = _env_float("TOPIC_MIN_FEASIBILITY_SCORE", 3.4)
TOPIC_JUDGE_PERSPECTIVES = ("research_significance", "technical_reader", "skeptical_editor")

# 选题编辑偏好：客观信号打完分后，再按「镜头」乘权重。
# 本刊以模型 / 算法 / 方法论为主；系统与工具链次之；产品 / 产业 / 政策可出现但不主导。
TOPIC_LENS_WEIGHTS = {
    "models_algorithms": 1.0,  # 模型、算法、训练、推理、评测、方法论
    "systems_infra": 0.92,  # Agent 系统、工具链、基础设施、安全机制
    "product_industry": 0.55,  # 产品发布、行业落地、商业案例
    "policy_society": 0.5,  # 政策监管、劳动力与社会影响
    "other": 0.7,
}
