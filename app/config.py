"""Runtime configuration, read from environment variables and validated at import.

Invalid values fail fast at startup instead of surfacing later as a division
by zero or a job that can never finish.
"""
import os
from pathlib import Path


class ConfigError(RuntimeError):
    """Raised when an environment variable holds an unusable value."""


def _int_env(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError:
        raise ConfigError(f'{name} must be an integer, got {raw!r}') from None
    if value < minimum:
        raise ConfigError(f'{name} must be >= {minimum}, got {value}')
    return value


REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
WHISPER_MODEL = os.getenv('WHISPER_MODEL', 'base')
UPLOAD_DIR = Path(os.getenv('UPLOAD_DIR', '/tmp/transcription_uploads')).resolve()

# Shared secret for the X-API-Key header. Checked by the API at startup (the
# worker does not need it).
API_KEY = os.getenv('API_KEY', '')

MAX_UPLOAD_MB = _int_env('MAX_UPLOAD_MB', 100)
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_AUDIO_SECONDS = _int_env('MAX_AUDIO_SECONDS', 3600)
# Maximum audio decoded per Whisper call. Chunks end at a segment boundary, so
# most are a little shorter than this.
CHUNK_SECONDS = _int_env('CHUNK_SECONDS', 60, minimum=10)
FFPROBE_TIMEOUT_SECONDS = _int_env('FFPROBE_TIMEOUT_SECONDS', 30)
FFMPEG_TIMEOUT_SECONDS = _int_env('FFMPEG_TIMEOUT_SECONDS', 600)

# How long job metadata, checkpoints and results are kept in Redis.
JOB_TTL_SECONDS = _int_env('JOB_TTL_SECONDS', 86400)
# Per-job lock lease. Renewed after every chunk, so it only has to outlast the
# processing of a single chunk, not the whole job.
LOCK_TIMEOUT_SECONDS = _int_env('LOCK_TIMEOUT_SECONDS', 900)
# Redis broker redelivers an unacknowledged task after this long. It must be
# longer than the slowest job, otherwise a running job is delivered twice (the
# per-job lock then stops the duplicate from doing any work).
VISIBILITY_TIMEOUT_SECONDS = _int_env('VISIBILITY_TIMEOUT_SECONDS', 7200)
# Uploads older than this with no active job are deleted by the cleanup task.
ORPHAN_UPLOAD_MIN_AGE_SECONDS = _int_env('ORPHAN_UPLOAD_MIN_AGE_SECONDS', 3600)

ALLOWED_EXTENSIONS = frozenset({'.wav', '.mp3', '.m4a', '.flac', '.ogg'})

if MAX_AUDIO_SECONDS > VISIBILITY_TIMEOUT_SECONDS:
    raise ConfigError('VISIBILITY_TIMEOUT_SECONDS must be at least MAX_AUDIO_SECONDS')

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
