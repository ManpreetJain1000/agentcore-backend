from celery import Celery

from app.core.config import settings


def create_celery_app() -> Celery:
    app = Celery(
        "agentcore",
        broker=settings.REDIS_URL,
        include=["app.tasks.agent_tasks"],
    )
    app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        timezone="UTC",
        enable_utc=True,
        task_ignore_result=True,          # state lives in Postgres, not the result backend
        task_acks_late=True,              # ack only after the task finishes
        task_reject_on_worker_lost=True,  # requeue if the worker process dies mid-run
        worker_prefetch_multiplier=1,     # long-running tasks: fetch one at a time
        broker_connection_retry_on_startup=True,
    )
    return app


celery_app = create_celery_app()