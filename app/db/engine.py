from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.settings import settings

engine = create_engine(
    settings.database_url,  # ✅ loaded from .env
    echo=True,  # 👈 logs SQL statements + values to stdout
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
