from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional


def _strtobool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    value = value.strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    return default


def _guess_project_root() -> Path:
    # agent_system_a/app/config.py -> .../agent_system_a/app -> .../agent_system_a -> .../repo_root
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "data").exists():
            return parent
    # Fall back to the common layout.
    return p.parents[3] if len(p.parents) > 3 else p.parent


def _get_required_path(env_var: str, default_path: Path) -> Path:
    raw = os.getenv(env_var)
    return Path(raw) if raw else default_path


def _build_redis_url_from_parts(
    host: str,
    port: str,
    db: str,
    password: str,
) -> str:
    if password:
        # redis://:password@host:port/db
        return f"redis://:{password}@{host}:{port}/{db}"
    return f"redis://{host}:{port}/{db}"


@dataclass(frozen=True)
class Settings:
    """
    Central configuration for the fitness coaching service.

    All values are environment-variable driven with sane Docker defaults.
    Import `Settings` (or `settings`) in other modules.
    """

    # Runtime / logging
    environment: str = os.getenv("ENV", "development")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    # Project-relative data paths (JSON inputs)
    project_root: Path = _guess_project_root()

    exercises_json_path: Path = _get_required_path(
        "EXERCISES_JSON_PATH",
        _guess_project_root() / "data" / "exercises.json",
    )
    workout_programs_json_path: Path = _get_required_path(
        "WORKOUT_PROGRAMS_JSON_PATH",
        _guess_project_root() / "data" / "workout_programs.json",
    )
    nutrition_guides_json_path: Path = _get_required_path(
        "NUTRITION_GUIDES_JSON_PATH",
        _guess_project_root() / "data" / "nutrition_guides.json",
    )

    # Service URLs (Docker defaults use service names from docker-compose.yml)

    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")

    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    # Redis: allow direct override, or compose from parts for convenience.
    redis_url: str = os.getenv(
        "REDIS_URL",
        _build_redis_url_from_parts(
            host=os.getenv("REDIS_HOST", "redis"),
            port=os.getenv("REDIS_PORT", "6379"),
            db=os.getenv("REDIS_DB", "0"),
            password=os.getenv("REDIS_PASSWORD", ""),
        ),
    )

    agent_system_b_url: str = os.getenv("AGENT_SYSTEM_B_URL", "http://agent-system-b:8001")
    mcp_server_url: str = os.getenv("MCP_SERVER_URL", "http://mcp-server:8002")

    # HTTP client behavior
    http_timeout_seconds: int = int(os.getenv("HTTP_TIMEOUT_SECONDS", "180"))
    http_max_connections: int = int(os.getenv("HTTP_MAX_CONNECTIONS", "100"))

    # Feature flags (optional; keep safe defaults)
    rag_auto_ingest: bool = _strtobool(os.getenv("RAG_AUTO_INGEST"), default=False)

    def as_dict(self) -> dict:
        # Convert Paths to strings for logging / serialization.
        return {
            "environment": self.environment,
            "log_level": self.log_level,
            "exercises_json_path": str(self.exercises_json_path),
            "workout_programs_json_path": str(self.workout_programs_json_path),
            "nutrition_guides_json_path": str(self.nutrition_guides_json_path),
            "ollama_url": self.ollama_url,
            "qdrant_url": self.qdrant_url,
            "redis_url": self.redis_url,
            "agent_system_b_url": self.agent_system_b_url,
            "mcp_server_url": self.mcp_server_url,
            "http_timeout_seconds": self.http_timeout_seconds,
            "http_max_connections": self.http_max_connections,
            "rag_auto_ingest": self.rag_auto_ingest,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


# Convenience singleton for modules that just need config.
settings = get_settings()

