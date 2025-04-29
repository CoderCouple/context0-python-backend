from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    clerk_api_key: str
    sql_alchemy_debug: bool = False
    db_host: str
    db_port: int
    db_user: str
    db_password: str
    db_name: str
    auth_disabled: bool = False
    cors_origins: list[str] = ["http://localhost:3000"]

    @property
    def database_url(self) -> str:
        return f"postgresql+psycopg://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"

    class Config:
        env_file = Path(__file__).resolve().parent.parent / ".env"


settings = Settings()
