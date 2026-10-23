import os
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

DEFAULT_DATABASE_URL = "sqlite:///analytics/sessions.db"


class Base(DeclarativeBase):
    pass


class LogEntry(Base):
    """One anonymized row per turn. No raw query text, no name, no session or
    user identifier of any kind — nothing here can be used to reconstruct a
    single person's conversation, matching CLAUDE.md's anonymity requirement."""

    __tablename__ = "log_entries"

    id: Mapped[int] = mapped_column(primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    role: Mapped[str] = mapped_column(String)
    course: Mapped[str | None] = mapped_column(String, nullable=True)
    topic_category: Mapped[str] = mapped_column(String)
    resource_type: Mapped[str | None] = mapped_column(String, nullable=True)
    escalated: Mapped[bool] = mapped_column(Boolean, default=False)
    had_results: Mapped[bool] = mapped_column(Boolean, default=False)


def get_engine(database_url: str | None = None):
    url = database_url or os.getenv("DATABASE_URL") or DEFAULT_DATABASE_URL
    engine = create_engine(url)
    Base.metadata.create_all(engine)
    return engine


def get_session(engine) -> Session:
    return Session(engine)
