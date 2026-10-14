# Import every model here so Alembic autogenerate can discover them.
from app.models.user import User

__all__ = ["User"]