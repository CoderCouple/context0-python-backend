import os
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Application environment
    app_env: str = "development"

    # Core settings
    clerk_secret_key: str = ""  # Make it optional with default
    sql_alchemy_debug: bool = False
    db_host: str
    db_port: int
    db_user: str
    db_password: str
    db_name: str
    auth_disabled: bool = False
    cors_origins: list[str] = ["http://localhost:3000"]
    clerk_webhook_signing_secret: str

    # MongoDB settings
    mongodb_connection_string: str = "mongodb://localhost:27017"
    mongodb_database_name: str = "context0"

    # LLM API Keys (optional - can be set per preset)
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    google_api_key: str = ""

    @property
    def database_url(self) -> str:
        return f"postgresql+psycopg://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"

    class Config:
        # Determine which .env file to use based on APP_ENV
        env = os.getenv("APP_ENV", "development")

        # Use .env.dev for development, .env.prod for production
        if env == "production":
            env_file = Path(__file__).resolve().parent.parent / ".env.prod"
        else:
            env_file = Path(__file__).resolve().parent.parent / ".env.dev"

        # Fallback to .env if specific file doesn't exist
        if not env_file.exists():
            env_file = Path(__file__).resolve().parent.parent / ".env"

        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
