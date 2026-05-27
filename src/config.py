from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, EnvSettingsSource, SettingsConfigDict

MODELS_DIR = Path("models")
DATA_DIR = Path("data")


class _CsvEnvSource(EnvSettingsSource):
    """Custom env source that parses comma-separated strings for list fields.

    Bypasses pydantic-settings' json.loads() path for these fields so that
    LEAGUES=mlb or ADMIN_IDS=123,456 work without wrapping in JSON brackets,
    and empty values fall back to their defaults.
    """

    _LIST_FIELDS: dict = {
        "leagues": (str, ["mlb"]),
        "admin_ids": (int, []),
    }

    def prepare_field_value(self, field_name, field, value, value_is_complex):
        if field_name in self._LIST_FIELDS and isinstance(value, str):
            elem_type, default = self._LIST_FIELDS[field_name]
            stripped = value.strip()
            if not stripped:
                return default
            # Accept both JSON array and comma-separated plain strings
            if stripped.startswith("["):
                return super().prepare_field_value(field_name, field, value, value_is_complex)
            try:
                return [elem_type(x.strip()) for x in stripped.split(",") if x.strip()]
            except (ValueError, TypeError):
                return default
        return super().prepare_field_value(field_name, field, value, value_is_complex)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True)

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
    min_edge: float = 0.03
    min_confidence: float = 0.55
    min_odds: float = 1.50
    max_odds: float = 4.50
    total_line: float = 8.5
    rl_line: float = 1.5
    itb_line: float = 4.5
    ai_ensemble_weight: float = 0.3
    ai_ensemble_top_n: int = 10
    ai_ensemble_min_prob: float = 0.55
    tz: str = "Europe/Moscow"

    def __init__(self, **data):
        super().__init__(**data)
        MODELS_DIR.mkdir(exist_ok=True)
        DATA_DIR.mkdir(exist_ok=True)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings=None,
        env_settings=None,
        dotenv_settings=None,
        secrets_settings=None,
        file_secret_settings=None,
        **_,
    ):
        extras = tuple(x for x in (dotenv_settings, secrets_settings, file_secret_settings) if x is not None)
        return (init_settings, _CsvEnvSource(settings_cls)) + extras


settings = Settings()
