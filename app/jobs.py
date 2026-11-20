"""Job registry and checkpoint storage in Redis.

Celery reports PENDING for any task ID it has never seen, so the API cannot use
Celery state alone to tell a queued job from a made-up ID. Every accepted upload
is therefore registered here first, and unknown IDs return 404.
"""
import json
import time
import uuid

import redis

from app.config import JOB_TTL_SECONDS, REDIS_URL

redis_client = redis.Redis.from_url(
    REDIS_URL, decode_responses=True,
    socket_connect_timeout=5, socket_timeout=30, health_check_interval=30,
)


def job_key(job_id: str) -> str:
    return f'transcription:job:{job_id}'


def lock_key(job_id: str) -> str:
    return f'transcription:lock:{job_id}'


def progress_key(job_id: str) -> str:
    return f'transcription:progress:{job_id}'


def segments_key(job_id: str) -> str:
    return f'transcription:segments:{job_id}'


def is_valid_job_id(job_id: str) -> bool:
    try:
        return str(uuid.UUID(job_id)) == job_id
    except (ValueError, TypeError):
        return False


def register_job(job_id: str, stored_name: str, original_filename: str) -> None:
    key = job_key(job_id)
    with redis_client.pipeline() as pipe:
        pipe.hset(key, mapping={
            'stored_name': stored_name,
            'filename': original_filename,
            'created_at': f'{time.time():.3f}',
        })
        pipe.expire(key, JOB_TTL_SECONDS)
        pipe.execute()


def get_job(job_id: str) -> dict | None:
    return redis_client.hgetall(job_key(job_id)) or None


def delete_job(job_id: str) -> None:
    redis_client.delete(job_key(job_id))


def is_terminal(job: dict) -> bool:
    return 'completed_at' in job or 'error' in job


def mark_completed(job_id: str) -> None:
    redis_client.hset(job_key(job_id), 'completed_at', f'{time.time():.3f}')


def mark_failed(job_id: str, error: str) -> None:
    redis_client.hset(job_key(job_id), mapping={'error': error, 'failed_at': f'{time.time():.3f}'})


# --- checkpoints -----------------------------------------------------------
# One list entry per finished chunk, so each checkpoint write is bounded by the
# size of one chunk's segments instead of rewriting the whole transcript.

def load_checkpoint(job_id: str, plan: dict) -> tuple[list[list[dict]], float, str | None]:
    """Return (per-chunk segments, resume position in seconds, language).

    The checkpoint is discarded if it was made for a different chunk size or
    duration (for example after a config change) or is malformed.
    """
    raw_meta = redis_client.get(progress_key(job_id))
    if raw_meta is None:
        return [], 0.0, None
    try:
        meta = json.loads(raw_meta)
        chunks = [json.loads(item) for item in redis_client.lrange(segments_key(job_id), 0, -1)]
        position = float(meta['next_start'])
        valid = (
            meta.get('plan') == plan
            and len(chunks) == meta.get('completed_chunks')
            and 0 <= position <= plan['duration'] + 0.01
            and all(isinstance(chunk, list) for chunk in chunks)
        )
    except (ValueError, TypeError, KeyError):
        valid = False
    if not valid:
        clear_checkpoint(job_id)
        return [], 0.0, None
    return chunks, position, meta.get('language')


def save_chunk(job_id: str, plan: dict, chunk_segments: list[dict], completed_chunks: int,
               next_start: float, language: str | None) -> None:
    """Append one chunk's segments and the resume position in a single transaction."""
    with redis_client.pipeline(transaction=True) as pipe:
        pipe.rpush(segments_key(job_id), json.dumps(chunk_segments))
        pipe.set(progress_key(job_id), json.dumps({
            'plan': plan, 'completed_chunks': completed_chunks,
            'next_start': next_start, 'language': language,
        }))
        pipe.expire(segments_key(job_id), JOB_TTL_SECONDS)
        pipe.expire(progress_key(job_id), JOB_TTL_SECONDS)
        pipe.execute()


def clear_checkpoint(job_id: str) -> None:
    redis_client.delete(progress_key(job_id), segments_key(job_id))
