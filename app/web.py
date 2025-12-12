# app/web.py
# Backward compatibility - imports from new API structure
from app.api.routes import app

__all__ = ["app"]
