from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader
from backend.app.core.config import settings

api_key_header = APIKeyHeader(name="X-Admin-Key", auto_error=False)


def verify_admin_key(api_key: str = Security(api_key_header)) -> bool:
    """Validate optional admin key for privileged administrative actions."""
    if not settings.admin_api_key:
        return True
    if api_key != settings.admin_api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing administrative API key.",
        )
    return True


def sanitize_input(text: str) -> str:
    """Strip dangerous control characters while preserving standard punctuation."""
    if not isinstance(text, str):
        return ""
    # Strip null bytes and non-printable control characters except newline and tab
    cleaned = "".join(ch for ch in text if ch in ("\n", "\r", "\t") or ch.isprintable())
    return cleaned.strip()
