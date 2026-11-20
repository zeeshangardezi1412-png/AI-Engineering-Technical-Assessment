# Audio Transcription Pipeline

A FastAPI service that accepts WAV, MP3, M4A, FLAC and OGG audio and returns Whisper speech-to-text results with per-segment timestamps. Celery and Redis run the jobs in the background, with progress, retries and resumable checkpoints.

```
client ──POST /transcriptions──▶ FastAPI ──validate (size, ffprobe)──▶ upload dir
                                    │                                     │
                                    └── register job + enqueue ──▶ Redis ─▶ Celery worker
                                                                             │ ffmpeg: ≤60 s chunks cut between segments
client ◀── GET status / result ── FastAPI ◀── job state, checkpoints ────────┘ Whisper per chunk
```

## Quick start (Docker, recommended)

Requires Docker with Compose v2.24+.

```bash
cp .env.example .env
# Set API_KEY in .env, e.g. python -c "import secrets; print(secrets.token_urlsafe(32))"
docker compose up -d --build
```

Open http://localhost:8000, paste the API key, drop in an audio file and select **Transcribe**.

Compose starts Redis (append-only persistence, bound to 127.0.0.1), the API on `127.0.0.1:${API_PORT:-8000}` and one worker. To test from another device on the same network, set `API_BIND=0.0.0.0` in `.env`, run `docker compose up -d`, and open `http://<this machine's IP>:8000`. Traffic is plain HTTP, so use this on a trusted network only. Whisper's `base` model downloads into a volume on the first job. Run more workers with `docker compose up -d --scale worker=2`.

## Run without Docker

Requires Python 3.10+, FFmpeg and FFprobe on `PATH`, and Redis (`docker compose up -d redis`).

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export API_KEY=change-me UPLOAD_DIR=/tmp/transcription_uploads   # same UPLOAD_DIR for API and worker

celery -A app.tasks.celery_app worker --loglevel=info --concurrency=1   # terminal 1
uvicorn app.main:app --reload                                          # terminal 2
```

On Windows, Celery's default pool does not work: add `--pool=solo`, or run the worker in Docker/WSL2. Both processes check at startup that `ffmpeg`/`ffprobe` can execute and exit with a clear error otherwise (Windows Smart App Control, for example, can block unsigned FFmpeg builds).

## Web UI

`GET /` serves a small page (plain HTML, CSS and JS in `app/static/`, no build step) that uses the same REST API:

- upload with progress, then live job status (queued, chunk progress, retries, failures with the server's reason)
- transcript with timestamps; selecting a segment plays the audio from that point, and the current segment is highlighted during playback
- copy the text or download it as SRT subtitles or JSON

The page is served with a strict Content-Security-Policy (only its own files, no inline scripts). Transcript text is inserted as text, never as HTML. The API key stays in `sessionStorage` for the current tab only.

## API

Every `/transcriptions` endpoint requires the `X-API-Key` header. Interactive docs: http://localhost:8000/docs (use **Authorize**).

| Method and path | Responses |
| --- | --- |
| `POST /transcriptions` (multipart field `file`) | `202` job queued · `400` unsupported extension, empty or undecodable file · `401` bad key · `413` over `MAX_UPLOAD_MB` or `MAX_AUDIO_SECONDS` · `503` Redis/queue unavailable |
| `GET /transcriptions/{job_id}` | `200` with `status`: `pending`, `processing` (with `progress`), `retrying`, `completed` or `failed` (with `error`) · `404` unknown job |
| `GET /transcriptions/{job_id}/result` | `200` transcript · `202` not finished yet · `422` job failed · `404` unknown job |
| `GET /health` | `200` liveness check, no key needed |

```bash
KEY=your-api-key
curl -H "X-API-Key: $KEY" -F "file=@sample.mp3" http://localhost:8000/transcriptions
curl -H "X-API-Key: $KEY" http://localhost:8000/transcriptions/JOB_ID
curl -H "X-API-Key: $KEY" http://localhost:8000/transcriptions/JOB_ID/result
```

Completed result:

```json
{
  "filename": "sample.mp3",
  "language": "en",
  "transcription": "Hello everyone. Welcome to the meeting.",
  "segments": [
    {"start": 0.0, "end": 1.8, "text": "Hello everyone."},
    {"start": 1.8, "end": 3.5, "text": "Welcome to the meeting."}
  ]
}
```

## Configuration

All settings are environment variables (see `.env.example`). Invalid values stop the process at startup.

| Variable | Default | Purpose |
| --- | --- | --- |
| `API_KEY` | none (required) | Shared secret for `X-API-Key` |
| `REDIS_URL` | `redis://localhost:6379/0` | Broker, result backend, job registry |
| `WHISPER_MODEL` | `base` | Any Whisper model name |
| `UPLOAD_DIR` | `/tmp/transcription_uploads` | Must be shared by API and worker |
| `MAX_UPLOAD_MB` / `MAX_AUDIO_SECONDS` | `100` / `3600` | Upload size and duration limits |
| `CHUNK_SECONDS` | `60` | Longest audio window per Whisper call (minimum 10) |
| `FFPROBE_TIMEOUT_SECONDS` / `FFMPEG_TIMEOUT_SECONDS` | `30` / `600` | Kill hung subprocesses |
| `JOB_TTL_SECONDS` | `86400` | Retention of job metadata, checkpoints and results |
| `LOCK_TIMEOUT_SECONDS` | `900` | Per-job lock lease, renewed after every chunk |
| `VISIBILITY_TIMEOUT_SECONDS` | `7200` | When Redis redelivers an unacknowledged task; must exceed the slowest job |
| `ORPHAN_UPLOAD_MIN_AGE_SECONDS` | `3600` | Age before an upload with no active job is deleted |

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The suite uses an in-memory Redis and a fake Whisper model, so it needs neither Redis nor a GPU. Two tests also use the real `ffprobe` when it is available.

## Assessment mapping

| Question | Solution |
| --- | --- |
| 4 | FastAPI multipart upload endpoint with size, extension and content (ffprobe) validation |
| 5 | Local open-source Whisper model, language auto-detected |
| 6 | Per-segment start/end timestamps in JSON |
| 7 | FFmpeg decodes every format to 16 kHz mono; FFprobe rejects non-audio before queueing |
| 8 | Chunks of up to 60 seconds that end at a Whisper segment boundary, with timestamps shifted to the full recording |
| 9 | Celery + Redis queue, configurable worker concurrency, per-job lock |
| 10 | Up to three retries with exponential backoff (5, 15, 45 s) for transient errors only; per-chunk checkpoints so retries and crash recovery resume |
| 11 | REST endpoints with interactive OpenAPI documentation, plus a small web UI on top of them |
| 12 | Push this project to your own GitHub repository |

## Design notes

- **Failure handling.** Invalid input (undecodable, no audio stream, too long, missing source file) fails at once with a readable `error`. Other errors are retried three times, then the job fails. Tasks are acknowledged only after they finish (`acks_late`, `reject_on_worker_lost`), so a crashed worker's job is redelivered and resumes from its last finished chunk.
- **Duplicate execution.** A Redis lock per job, renewed after every chunk, means only one worker processes a job. A duplicate delivery waits for the lease to expire (covering a crashed holder) and is dropped if the job is still owned or already finished. A worker that loses its lease stops without writing.
- **Chunk boundaries.** A fixed 60 s cut splits words, and overlapping windows duplicate them because Whisper segments (often 5–10 s) are longer than any small overlap. Instead, the worker drops each window's last segment, which is usually cut off at the edge, and starts the next window where that segment began. This is the approach Whisper uses internally for long audio. Every stretch of speech is transcribed once, and boundaries fall between segments. A window holding one very long segment falls back to a hard cut, so the job always advances by at least half a chunk.
- **Cleanup.** Uploads are deleted when a job completes or fails permanently. An hourly task (and one on worker start) deletes uploads whose job is finished, expired or was never registered.
- **Worker input.** The task receives only the stored file name (`<job uuid>.<ext>`), never a path, so a forged queue message cannot make the worker read or delete other files.

## Known limitations

- One shared API key: no per-user accounts or job ownership. Anyone holding the key can read any job. Add real authentication (OAuth/JWT) and store an owner per job before exposing the service.
- No rate limiting or quotas. Put the API behind a gateway that provides them. The request body is still received (up to the size cap) before the API key is checked.
- API and worker share a filesystem. Multi-host deployments need object storage (S3 or similar).
- Redis holds the queue, job state and results. Use a managed Redis with authentication, TLS and backups, or move results to a database.
- Crash recovery waits for `VISIBILITY_TIMEOUT_SECONDS` before Redis redelivers the task.
- FFprobe/FFmpeg parse untrusted media. Keep them patched and run workers with CPU/memory limits. There is no antivirus scanning.
- No dead-letter queue or alerting: failed jobs are visible through the status endpoint and worker logs.
