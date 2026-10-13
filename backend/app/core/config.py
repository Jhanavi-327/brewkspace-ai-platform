from pathlib import Path

from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict
)


BASE_DIR = Path(__file__).resolve().parent.parent.parent

ENV_FILE = BASE_DIR / ".env"


class Settings(BaseSettings):

    # Database
    DATABASE_URL: str

    # JWT
    SECRET_KEY: str
    ALGORITHM: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int

    # Google Gemini
    GOOGLE_API_KEY: str | None = None

    # Patta Book: run a second, different Gemini model as a verification pass and
    # flag rows where the two disagree on text fields (holder/father/survey). This
    # is the text analogue of the numeric checksum. Best-effort — if no second
    # model is available (quota), verification is skipped and the primary result
    # is unaffected. Costs one extra Gemini call per Patta doc when enabled.
    PATTA_VERIFY_TEXT_FIELDS: bool = True

    # Patta Book self-repair: Gemini at temperature=0 is NOT deterministic, so the
    # same doc can extract cleanly on one call and fabricate numbers on the next.
    # When an extraction's entry columns don't reconcile to the printed grand-total
    # row, re-run on the next model and accept the first read that reconciles.
    # This turns the intermittent fabrication into a self-correcting loop. Bounded
    # by PATTA_RECONCILE_MAX_ATTEMPTS (each costs one Gemini call, only spent when a
    # doc fails to reconcile). Set False to accept the first successful extraction.
    PATTA_RECONCILE_RETRY: bool = True
    PATTA_RECONCILE_MAX_ATTEMPTS: int = 3

    # Mutation Register (ನಮೂನೆ-12 / ಮ್ಯುಟೇಶನ್ ರಿಜಿಸ್ಟರ್). Unlike Patta there is
    # no printed arithmetic total to reconcile, so trust rests on a second,
    # different Gemini model read: cross-check the header/entry text fields and
    # flag any where the two reads disagree. Best-effort — if no second model is
    # available (quota) verification is skipped and the primary result stands.
    # Costs one extra Gemini call per MR doc only when no spare read from the
    # quota-fallback loop is already available.
    MR_VERIFY_TEXT_FIELDS: bool = True

    # If the first model's read is incomplete (missing core header fields or no
    # transaction row), re-run on the next model up to this many total reads and
    # keep the fullest. Set MR_VERIFY_RETRY False to accept the first read.
    MR_VERIFY_RETRY: bool = True
    MR_VERIFY_MAX_ATTEMPTS: int = 3

    # LangSmith tracing. Read through Settings like everything else so it
    # resolves from the same .env; llm_usage then exports these into the
    # process environment, which is where the LangSmith SDK looks for them.
    LANGSMITH_TRACING: str | None = None
    LANGSMITH_API_KEY: str | None = None
    LANGSMITH_PROJECT: str | None = None
    LANGSMITH_ENDPOINT: str | None = None

    # Per-million-token Gemini rates as JSON, for per-document cost reporting.
    GEMINI_PRICES: str | None = None
    GEMINI_PRICE_CURRENCY: str = "USD"

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()