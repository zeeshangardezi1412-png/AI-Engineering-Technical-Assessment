"""Test setup: in-memory Redis, a small upload limit and a fake Whisper model.

These tests do not need Redis, FFmpeg or Whisper. Tests that call the real
ffprobe binary are skipped when it is unavailable.
"""
import io
import math
import os
import shutil
import struct
import subprocess
import tempfile
import wave
from pathlib import Path

# Configuration is read at import time, so set it before importing the app.
os.environ.update({
    'API_KEY': 'test-api-key',
    'UPLOAD_DIR': tempfile.mkdtemp(prefix='transcription-tests-'),
    'MAX_UPLOAD_MB': '1',
    'MAX_AUDIO_SECONDS': '600',
    'CHUNK_SECONDS': '60',
    'REDIS_URL': 'redis://localhost:6390/0',  # never contacted: replaced below
})

import fakeredis  # noqa: E402
import pytest  # noqa: E402

from app import jobs, tasks  # noqa: E402
from app.config import UPLOAD_DIR  # noqa: E402

tasks.celery_app.conf.result_backend = 'cache+memory://'
API_KEY = os.environ['API_KEY']


def _ffprobe_works() -> bool:
    if not shutil.which('ffprobe'):
        return False
    try:
        subprocess.run(['ffprobe', '-version'], capture_output=True, check=True, timeout=15)
        return True
    except (OSError, subprocess.SubprocessError):
        return False


requires_ffprobe = pytest.mark.skipif(not _ffprobe_works(), reason='ffprobe not available')


def make_wav(seconds: float = 1.0, rate: int = 16000) -> bytes:
    """A short mono 16-bit sine-wave WAV file."""
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        frames = int(seconds * rate)
        wav.writeframes(b''.join(
            struct.pack('<h', int(8000 * math.sin(2 * math.pi * 440 * i / rate))) for i in range(frames)))
    return buffer.getvalue()


@pytest.fixture(autouse=True)
def fake_redis(monkeypatch):
    client = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(jobs, 'redis_client', client)
    yield client
    client.flushall()


@pytest.fixture(autouse=True)
def clean_upload_dir():
    yield
    for path in Path(UPLOAD_DIR).iterdir():
        path.unlink()


class FakeWhisper:
    """Returns one segment per 10 s of decoded window, labelled with its absolute window time."""

    def __init__(self):
        self.windows = []        # (start, duration) of each decoded chunk, in call order
        self.fail_on_call = {}   # call number -> exception to raise

    def record_window(self, start, duration):
        self.windows.append((start, duration))

    def transcribe(self, path, language=None, fp16=False):
        call = len(self.windows) - 1
        if call in self.fail_on_call:
            raise self.fail_on_call.pop(call)
        start, duration = self.windows[-1]
        segments = []
        offset = 0.0
        while offset < duration:
            end = min(offset + 10, duration)
            segments.append({'start': offset, 'end': end, 'text': f' t{start + offset:.0f} '})
            offset = end
        return {'language': language or 'en', 'segments': segments}
