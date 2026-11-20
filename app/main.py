"""HTTP API: accepts audio uploads, queues transcription jobs and reports their status."""
import logging
import secrets
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from celery.result import AsyncResult
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from kombu.exceptions import OperationalError as BrokerError
from redis.exceptions import RedisError

from app import jobs
from app.audio import AudioTooLongError, InvalidAudioError, check_binaries, probe_audio
from app.config import ALLOWED_EXTENSIONS, API_KEY, MAX_UPLOAD_BYTES, MAX_UPLOAD_MB, UPLOAD_DIR
from app.tasks import celery_app, transcribe_audio

logger = logging.getLogger(__name__)

# Allowance for multipart framing (boundary lines, part headers) around the file.
MULTIPART_OVERHEAD_BYTES = 64 * 1024
MAX_FILENAME_LENGTH = 255
STATIC_DIR = Path(__file__).parent / 'static'
# The web UI loads only its own files; transcript text is inserted as text, never HTML.
UI_HEADERS = {
    'Content-Security-Policy': (
        "default-src 'self'; img-src 'self' data:; media-src 'self' blob:; object-src 'none'; "
        "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"),
    'X-Content-Type-Options': 'nosniff',
    'Referrer-Policy': 'no-referrer',
}
STATES = {
    'PENDING': 'pending', 'STARTED': 'processing', 'PROGRESS': 'processing',
    'SUCCESS': 'completed', 'FAILURE': 'failed', 'RETRY': 'retrying',
}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if not API_KEY:
        raise RuntimeError('API_KEY must be set (see .env.example)')
    check_binaries('ffprobe')
    yield


app = FastAPI(title='Audio Transcription Pipeline', version='1.1.0', lifespan=lifespan)
app.mount('/static', StaticFiles(directory=STATIC_DIR), name='static')
api_key_header = APIKeyHeader(name='X-API-Key', auto_error=False)


def require_api_key(api_key: str | None = Depends(api_key_header)) -> None:
    if not api_key or not secrets.compare_digest(api_key.encode(), API_KEY.encode()):
        raise HTTPException(401, 'Invalid or missing API key')


@app.exception_handler(RedisError)
@app.exception_handler(BrokerError)
async def backend_unavailable(_request: Request, exc: Exception):
    logger.error('Job store unavailable: %r', exc)
    return JSONResponse(status_code=503, content={'detail': 'Job store unavailable; try again later'})


@app.get('/', include_in_schema=False)
def web_ui():
    return FileResponse(STATIC_DIR / 'index.html', headers=UI_HEADERS)


@app.get('/health')
def health():
    return {'status': 'ok'}


async def _save_upload(file: UploadFile, destination: Path) -> int:
    size = 0
    with destination.open('xb') as out:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                raise HTTPException(413, f'Audio file exceeds the {MAX_UPLOAD_MB} MB upload limit')
            out.write(chunk)
    return size


def _discard_job(job_id: str) -> None:
    try:
        jobs.delete_job(job_id)
    except RedisError:
        pass


@app.post('/transcriptions', status_code=202, dependencies=[Depends(require_api_key)],
          max_body_size=MAX_UPLOAD_BYTES + MULTIPART_OVERHEAD_BYTES)
async def create_transcription(file: UploadFile = File(...)):
    original_name = Path(file.filename or '').name[:MAX_FILENAME_LENGTH]
    extension = Path(original_name).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        await file.close()
        raise HTTPException(400, 'Unsupported format. Use WAV, MP3, M4A, FLAC or OGG.')

    job_id = str(uuid.uuid4())
    stored_name = f'{job_id}{extension}'
    path = UPLOAD_DIR / stored_name
    try:
        if await _save_upload(file, path) == 0:
            raise HTTPException(400, 'Audio file is empty')
        try:
            # Check the content, not just the extension, before it reaches the queue.
            await run_in_threadpool(probe_audio, path)
        except AudioTooLongError as exc:
            raise HTTPException(413, str(exc)) from None
        except InvalidAudioError as exc:
            raise HTTPException(400, str(exc)) from None
        try:
            jobs.register_job(job_id, stored_name, original_name)
            transcribe_audio.apply_async(args=[stored_name, original_name], task_id=job_id)
        except (RedisError, BrokerError):
            logger.exception('Could not queue job %s', job_id)
            _discard_job(job_id)
            raise HTTPException(503, 'Transcription queue unavailable; try again later') from None
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    finally:
        await file.close()
    return {'job_id': job_id, 'status': 'pending', 'status_url': f'/transcriptions/{job_id}'}


def _load_job(job_id: str) -> dict:
    job = jobs.get_job(job_id) if jobs.is_valid_job_id(job_id) else None
    if job is None:
        raise HTTPException(404, 'Job not found')
    return job


@app.get('/transcriptions/{job_id}', dependencies=[Depends(require_api_key)])
def get_status(job_id: str):
    job = _load_job(job_id)
    task = AsyncResult(job_id, app=celery_app)
    response = {'job_id': job_id, 'status': STATES.get(task.state, task.state.lower())}
    if task.state == 'PROGRESS' and isinstance(task.info, dict):
        response['progress'] = task.info
    if task.state == 'FAILURE':
        response['error'] = job.get('error', 'Transcription failed; inspect worker logs.')
    if task.state == 'SUCCESS':
        response['result_url'] = f'/transcriptions/{job_id}/result'
    return response


@app.get('/transcriptions/{job_id}/result', dependencies=[Depends(require_api_key)])
def get_result(job_id: str):
    job = _load_job(job_id)
    task = AsyncResult(job_id, app=celery_app)
    if task.state == 'FAILURE':
        raise HTTPException(422, job.get('error', 'Transcription failed'))
    if task.state != 'SUCCESS':
        return JSONResponse(status_code=202, content={
            'job_id': job_id, 'status': STATES.get(task.state, task.state.lower()),
            'detail': 'Transcription not completed yet',
        })
    return task.result
