from pathlib import Path
from typing import Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings

MODELS_DIR = Path("models")
DATA_DIR = Path("data")


class Settings(BaseSettings):
    model_config = {"env_file": ".env"}

    telegram_bot_token: str = ""
    admin_ids: list[int] = []
    mlb_sport_id: int = 1
    odds_api_key: str = ""
    anthropic_api_key: Optional[str] = None
    llm_base_url: Optional[str] = None
    llm_api_key: Optional[str] = None
    llm_model: str = "claude-3-5-sonnet-20241022"
    tavily_api_key: Optional[str] = None
    leagues: list[str] = ["mlb"]
    database_url: Optional[str] = None
    min_edge: float = 0.05
    min_confidence: float = 0.55
    min_odds: float = 1.50
    max_odds: float = 4.50
    total_line: float = 8.5
    rl_line: float = 1.5
    ai_ensemble_weight: float = 0.3
    ai_ensemble_top_n: int = 10
    ai_ensemble_min_prob: float = 0.55
    tz: str = "Europe/Moscow"

    def __init__(self, **data):
        super().__init__(**data)
        MODELS_DIR.mkdir(exist_ok=True)
        DATA_DIR.mkdir(exist_ok=True)

    @field_validator("admin_ids", mode="before")
    @classmethod
    def parse_admin_ids(cls, v):
        if isinstance(v, str):
            return [int(x.strip()) for x in v.split(",") if x.strip()]
        return v or []

    @field_validator("leagues", mode="before")
    @classmethod
    def parse_leagues(cls, v):
        if isinstance(v, str):
            if not v.strip():
                return ["mlb"]
            return [x.strip().lower() for x in v.split(",") if x.strip()]
        return v or ["mlb"]


settings = Settings()
