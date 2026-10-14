from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status

from app.agent.memory_manager import LongTermMemoryStore
from app.core.config import settings
from app.core.dependencies import LLM, CurrentUser, DbSession
from app.models.memory import LongTermMemory
from app.schemas.agent import MemoryOut, MemorySearchResult

router = APIRouter(prefix="/memory", tags=["memory"])


@router.get(
    "",
    response_model=list[MemoryOut],
    summary="List the caller's long-term memories (newest first)",
    responses={401: {"description": "Missing or invalid token"}},
)
async def list_memories(
    current_user: CurrentUser,
    db: DbSession,
    llm: LLM,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[LongTermMemory]:
    store = LongTermMemoryStore(db, llm)
    return await store.list_for_user(current_user.id, limit, offset)


@router.get(
    "/search",
    response_model=list[MemorySearchResult],
    summary="Semantic search over the caller's long-term memories",
    responses={
        401: {"description": "Missing or invalid token"},
        503: {"description": "Embedding service unavailable"},
    },
)
async def search_memories(
    current_user: CurrentUser,
    db: DbSession,
    llm: LLM,
    q: Annotated[str, Query(min_length=1, max_length=1000, description="Plain-text query")],
    limit: Annotated[int, Query(ge=1, le=20)] = settings.LONG_TERM_SEARCH_LIMIT,
    min_similarity: Annotated[
        float, Query(ge=-1.0, le=1.0, description="Hide results below this similarity")
    ] = 0.0,
) -> list[MemorySearchResult]:
    store = LongTermMemoryStore(db, llm)
    try:
        results = await store.search_with_scores(current_user.id, q, limit)
    except Exception:  # noqa: BLE001 - provider error, timeout, etc.
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "Embedding service unavailable, try again later"
        )

    response: list[MemorySearchResult] = []
    for memory, distance in results:
        similarity = round(1.0 - distance, 4)
        if similarity < min_similarity:
            continue
        response.append(
            MemorySearchResult(
                **MemoryOut.model_validate(memory).model_dump(),
                similarity=similarity,
            )
        )
    return response


@router.delete(
    "/{memory_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete one long-term memory",
    responses={
        401: {"description": "Missing or invalid token"},
        404: {"description": "Memory not found"},
    },
)
async def delete_memory(memory_id: str, current_user: CurrentUser, db: DbSession, llm: LLM) -> None:
    store = LongTermMemoryStore(db, llm)
    if not await store.delete(current_user.id, memory_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Memory not found")