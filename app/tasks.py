"""Celery worker: chunked Whisper transcription with checkpoints, retries and a per-job lock."""
import logging
import math
import re
import tempfile
import time
from pathlib import Path

from celery import Celery
from celery.exceptions import Ignore
from celery.signals import worker_init, worker_ready
from redis.exceptions import LockError, RedisError

from app import jobs
from app.audio import InvalidAudioError, check_binaries, convert_chunk, probe_audio
from app.config import (
    ALLOWED_EXTENSIONS,
    CHUNK_SECONDS,
    JOB_TTL_SECONDS,
    LOCK_TIMEOUT_SECONDS,
    ORPHAN_UPLOAD_MIN_AGE_SECONDS,
    REDIS_URL,
    UPLOAD_DIR,
    VISIBILITY_TIMEOUT_SECONDS,
    WHISPER_MODEL,
)

logger = logging.getLogger(__name__)

celery_app = Celery('transcription', broker=REDIS_URL, backend=REDIS_URL)
celery_app.conf.update(
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    task_track_started=True,
    # Acknowledge only after the task finishes, and requeue it if the worker process
    # dies, so a crash does not lose the job. Checkpoints let the rerun resume.
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    result_expires=JOB_TTL_SECONDS,
    broker_transport_options={'visibility_timeout': VISIBILITY_TIMEOUT_SECONDS},
    broker_connection_retry_on_startup=True,
    redis_socket_connect_timeout=5,
    redis_socket_timeout=30,
    beat_schedule={
        'cleanup-orphaned-uploads': {
            'task': 'app.tasks.cleanup_orphaned_uploads',
            'schedule': 3600.0,
        },
    },
)

MAX_RETRIES = 3
_model = None


def get_model():
    global _model
    if _model is None:
        import whisper
        _model = whisper.load_model(WHISPER_MODEL)
    return _model


# Uploads are stored as "<job uuid><extension>"; nothing else is accepted.
_STORED_NAME = re.compile(
    r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}(%s)'
    % '|'.join(re.escape(ext) for ext in sorted(ALLOWED_EXTENSIONS)))


def upload_path(stored_name: str) -> Path:
    """Map a stored upload name to its file in UPLOAD_DIR.

    Task arguments come from the broker, so a forged message must not be able to
    make the worker read or delete files outside the upload directory.
    """
    if not isinstance(stored_name, str) or not _STORED_NAME.fullmatch(stored_name):
        raise InvalidAudioError('Invalid upload reference')
    return UPLOAD_DIR / stored_name


def _remove_upload(stored_name: str) -> None:
    try:
        upload_path(stored_name).unlink(missing_ok=True)
    except (InvalidAudioError, OSError):
        logger.warning('Could not remove upload %r', stored_name, exc_info=True)


def _fail(job_id: str, stored_name: str, error: str) -> None:
    """Record a terminal failure and free its resources."""
    try:
        jobs.mark_failed(job_id, error)
        jobs.clear_checkpoint(job_id)
    except RedisError:
        logger.exception('Could not record failure for job %s', job_id)
    _remove_upload(stored_name)


def _renew(lock) -> None:
    """Extend the lease; stop quietly if another worker has taken the job over."""
    try:
        lock.reacquire()
    except LockError:
        logger.warning('Lost lock for job; another worker owns it now')
        raise Ignore() from None


def _release(lock) -> None:
    try:
        lock.release()
    except LockError:
        pass  # Lease already expired or taken over.


def _window_segments(raw_segments: list[dict], window_start: float, window_end: float,
                     is_final: bool) -> tuple[list[dict], float]:
    """Shift Whisper segments to recording time and choose where the next window starts.

    Speech running past the window edge comes back as a truncated last segment. Unless
    this is the final window, that segment is dropped and the next window starts at
    its beginning, so every word is transcribed once and chunk boundaries fall
    between segments. If the last segment starts too early to make enough progress
    (one very long segment), the window is kept whole and cut at its edge instead.
    """
    segments = []
    for segment in raw_segments:
        text = str(segment.get('text', '')).strip()
        if not text:
            continue
        start = window_start + float(segment['start'])
        end = max(start, min(window_start + float(segment['end']), window_end))
        segments.append((start, end, text))

    next_start = window_end
    if not is_final and len(segments) > 1 and segments[-1][0] >= window_start + CHUNK_SECONDS / 2:
        next_start = segments.pop()[0]
    return [{'start': round(s, 2), 'end': round(e, 2), 'text': t} for s, e, t in segments], next_start


def _progress(completed_chunks: int, position: float, duration: float) -> dict:
    remaining_chunks = math.ceil(max(0.0, duration - position) / CHUNK_SECONDS)
    return {
        'completed_chunks': completed_chunks,
        'total_chunks': completed_chunks + remaining_chunks,  # estimate until the last chunk
        'processed_seconds': round(position, 2),
        'duration_seconds': round(duration, 2),
    }


def _transcribe(task, job_id: str, stored_name: str, original_filename: str, lock) -> dict:
    path = upload_path(stored_name)
    if not path.is_file():
        raise InvalidAudioError('Uploaded file is no longer available')

    duration = probe_audio(path)
    plan = {'chunk_seconds': CHUNK_SECONDS, 'duration': round(duration, 3)}
    done, position, language = jobs.load_checkpoint(job_id, plan)
    if done:
        logger.info('Job %s resuming from checkpoint at %.1fs after %d chunk(s)', job_id, position, len(done))

    model = get_model()
    _renew(lock)  # The first model load can take minutes (download).

    with tempfile.TemporaryDirectory() as temp_dir:
        while position < duration:
            window_end = min(position + CHUNK_SECONDS, duration)
            wav = Path(temp_dir) / f'chunk_{len(done)}.wav'
            convert_chunk(path, wav, position, window_end - position)
            output = model.transcribe(str(wav), language=language, fp16=False)
            wav.unlink(missing_ok=True)
            language = output.get('language') or language
            segments, next_start = _window_segments(
                output.get('segments', []), position, window_end, is_final=window_end >= duration)

            _renew(lock)  # Only the lock holder may write the checkpoint.
            jobs.save_chunk(job_id, plan, segments, len(done) + 1, next_start, language)
            done.append(segments)
            position = next_start
            task.update_state(state='PROGRESS', meta=_progress(len(done), position, duration))
            logger.info('Job %s chunk %d done; %.1f/%.1fs transcribed', job_id, len(done), position, duration)

    segments = [segment for chunk_segments in done for segment in chunk_segments]
    return {
        'filename': original_filename,
        'language': language,
        'transcription': ' '.join(segment['text'] for segment in segments),
        'segments': segments,
    }


@celery_app.task(bind=True, max_retries=MAX_RETRIES, name='app.tasks.transcribe_audio')
def transcribe_audio(self, stored_name: str, original_filename: str):
    job_id = self.request.id
    job = jobs.get_job(job_id)
    if job is None:
        logger.warning('Job %s is not registered (unknown or expired); dropping it', job_id)
        _remove_upload(stored_name)
        raise Ignore()
    if jobs.is_terminal(job):
        logger.info('Job %s already finished; ignoring duplicate delivery', job_id)
        raise Ignore()

    lock = jobs.redis_client.lock(jobs.lock_key(job_id), timeout=LOCK_TIMEOUT_SECONDS, blocking=False)
    if not lock.acquire():
        # Another worker is processing this job (or crashed while holding the lease).
        # Check again once the lease would have expired.
        if self.request.retries >= self.max_retries:
            logger.warning('Job %s still locked by another worker; dropping duplicate', job_id)
            raise Ignore()
        logger.info('Job %s is locked by another worker; checking again in %ss', job_id, LOCK_TIMEOUT_SECONDS)
        # Re-queue this delivery without marking the job as retrying: the job itself
        # is healthy, and its visible state belongs to the worker holding the lock.
        self.retry(countdown=LOCK_TIMEOUT_SECONDS, throw=False)
        raise Ignore()

    try:
        result = _transcribe(self, job_id, stored_name, original_filename, lock)
        jobs.mark_completed(job_id)
        jobs.clear_checkpoint(job_id)
    except Ignore:
        raise
    except InvalidAudioError as exc:
        # Permanent: retrying the same input cannot succeed.
        logger.warning('Job %s rejected: %s', job_id, exc)
        _fail(job_id, stored_name, str(exc))
        raise
    except Exception as exc:
        logger.exception('Transcription attempt %d failed for job %s', self.request.retries + 1, job_id)
        if self.request.retries >= self.max_retries:
            _fail(job_id, stored_name, 'Transcription failed after retries; see worker logs')
            raise
        raise self.retry(exc=exc, countdown=min(5 * 3 ** self.request.retries, 300))
    finally:
        _release(lock)

    _remove_upload(stored_name)
    return result


@celery_app.task(name='app.tasks.cleanup_orphaned_uploads')
def cleanup_orphaned_uploads() -> int:
    """Delete old uploads whose job is finished, expired or was never registered.

    Covers files left behind when a process is killed before its own cleanup runs.
    """
    cutoff = time.time() - ORPHAN_UPLOAD_MIN_AGE_SECONDS
    removed = 0
    for path in UPLOAD_DIR.iterdir():
        try:
            if not path.is_file() or path.stat().st_mtime > cutoff:
                continue
            job = jobs.get_job(path.stem) if jobs.is_valid_job_id(path.stem) else None
            if job is None or jobs.is_terminal(job):
                path.unlink(missing_ok=True)
                removed += 1
        except OSError:
            logger.warning('Could not inspect upload %s', path, exc_info=True)
    if removed:
        logger.info('Removed %d orphaned upload(s)', removed)
    return removed


@worker_init.connect
def _check_worker_dependencies(**_):
    check_binaries('ffmpeg', 'ffprobe')


@worker_ready.connect
def _cleanup_on_start(**_):
    try:
        cleanup_orphaned_uploads()
    except RedisError:
        logger.exception('Startup cleanup skipped: Redis unavailable')
