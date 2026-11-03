"""Carga de configuracion desde variables de entorno (.env)."""
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY: str | None = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
SQLITE_DB_PATH: str = os.getenv("SQLITE_DB_PATH", "checkpoints.sqlite")
RECURSION_LIMIT: int = int(os.getenv("RECURSION_LIMIT", "10"))
