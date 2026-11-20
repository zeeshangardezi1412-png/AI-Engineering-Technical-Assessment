import uuid
from pathlib import Path

import pytest
import redis
from fastapi.testclient import TestClient

from app import jobs, main
from app.audio import AudioTooLongError, InvalidAudioError
from app.config import MAX_UPLOAD_BYTES, UPLOAD_DIR
from conftest import API_KEY, make_wav, requires_ffprobe

AUTH = {'X-API-Key': API_KEY}


class FakeAsyncResult:
    states = {}  # job_id -> (state, info)

    def __init__(self, job_id, app=None):
        self.state, self.info = self.states.get(job_id, ('PENDING', None))
        self.result = self.info


@pytest.fixture
def queued(monkeypatch):
    """Capture queued tasks instead of publishing them."""
    calls = []
    monkeypatch.setattr(main.transcribe_audio, 'apply_async',
                        lambda args, task_id: calls.append((args, task_id)))
    return calls


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main, 'check_binaries', lambda *names: None)
    monkeypatch.setattr(main, 'AsyncResult', FakeAsyncResult)
    FakeAsyncResult.states = {}
    with TestClient(main.app) as test_client:
        yield test_client


@pytest.fixture
def probe_ok(monkeypatch):
    monkeypatch.setattr(main, 'probe_audio', lambda path: 1.0)


def upload(client, name='sample.wav', content=None, headers=AUTH):
    content = make_wav() if content is None else content
    return client.post('/transcriptions', files={'file': (name, content, 'audio/wav')}, headers=headers)


def uploads_on_disk():
    return list(Path(UPLOAD_DIR).iterdir())


def test_health_is_public(client):
    assert client.get('/health').json() == {'status': 'ok'}


def test_web_ui_is_served_with_strict_csp(client):
    page = client.get('/')
    assert page.status_code == 200 and 'text/html' in page.headers['content-type']
    assert "default-src 'self'" in page.headers['content-security-policy']
    assert page.headers['x-content-type-options'] == 'nosniff'
    for asset in ('/static/app.js', '/static/styles.css'):
        assert client.get(asset).status_code == 200
    assert 'innerHTML = segment' not in client.get('/static/app.js').text


@pytest.mark.parametrize('name', ['sample.wav', 'sample.MP3', 'talk.m4a', 'a.flac', 'b.ogg'])
def test_valid_upload_is_queued(client, queued, probe_ok, name):
    response = upload(client, name=name)
    assert response.status_code == 202
    body = response.json()
    assert body['status'] == 'pending'
    assert body['status_url'] == f"/transcriptions/{body['job_id']}"
    (stored_name, original), task_id = queued[0]
    assert task_id == body['job_id'] and original == name
    assert stored_name == f"{body['job_id']}{Path(name).suffix.lower()}"
    assert (Path(UPLOAD_DIR) / stored_name).is_file()
    assert jobs.get_job(body['job_id'])['filename'] == name


@pytest.mark.parametrize('headers', [{}, {'X-API-Key': 'wrong'}, {'X-API-Key': ''}])
def test_missing_or_wrong_api_key_is_rejected(client, queued, probe_ok, headers):
    job_id = str(uuid.uuid4())
    jobs.register_job(job_id, f'{job_id}.wav', 'x.wav')
    assert upload(client, headers=headers).status_code == 401
    assert client.get(f'/transcriptions/{job_id}', headers=headers).status_code == 401
    assert client.get(f'/transcriptions/{job_id}/result', headers=headers).status_code == 401
    assert queued == [] and uploads_on_disk() == []


@pytest.mark.parametrize('name', ['notes.txt', 'audio', 'archive.wav.exe', ''])
def test_unsupported_extension_is_rejected(client, queued, name):
    response = upload(client, name=name or 'x')
    assert response.status_code == 400
    assert queued == [] and uploads_on_disk() == []


def test_empty_file_is_rejected(client, queued):
    response = upload(client, content=b'')
    assert response.status_code == 400
    assert response.json()['detail'] == 'Audio file is empty'
    assert queued == [] and uploads_on_disk() == []


def test_body_over_limit_is_rejected_before_parsing(client, queued):
    response = upload(client, content=b'\0' * (MAX_UPLOAD_BYTES + main.MULTIPART_OVERHEAD_BYTES + 1))
    assert response.status_code == 413
    assert queued == [] and uploads_on_disk() == []


def test_file_over_limit_is_rejected_while_streaming(client, queued):
    # Small enough to pass the request-body limit, but the file itself is over the cap.
    response = upload(client, content=b'\0' * (MAX_UPLOAD_BYTES + 1024))
    assert response.status_code == 413
    assert 'upload limit' in response.json()['detail']
    assert queued == [] and uploads_on_disk() == []


def test_invalid_audio_is_rejected_by_probe(client, queued, monkeypatch):
    def reject(path):
        raise InvalidAudioError('File is not a decodable audio file')
    monkeypatch.setattr(main, 'probe_audio', reject)
    response = upload(client, content=b'definitely not audio')
    assert response.status_code == 400
    assert queued == [] and uploads_on_disk() == []


def test_too_long_audio_is_rejected(client, queued, monkeypatch):
    def too_long(path):
        raise AudioTooLongError('Audio is 7200s long; the limit is 600s')
    monkeypatch.setattr(main, 'probe_audio', too_long)
    response = upload(client)
    assert response.status_code == 413
    assert 'limit is 600s' in response.json()['detail']
    assert queued == [] and uploads_on_disk() == []


@requires_ffprobe
def test_real_ffprobe_accepts_wav_and_rejects_fake_mp3(client, queued):
    assert upload(client, content=make_wav(1.0)).status_code == 202
    response = upload(client, name='fake.mp3', content=b'ID3' + b'\x00' * 2000)
    assert response.status_code == 400
    assert len(queued) == 1 and len(uploads_on_disk()) == 1


@pytest.mark.parametrize('job_id', [str(uuid.uuid4()), 'not-a-uuid', '..%2F..%2Fetc'])
def test_unknown_job_returns_404(client, job_id):
    assert client.get(f'/transcriptions/{job_id}', headers=AUTH).status_code == 404
    assert client.get(f'/transcriptions/{job_id}/result', headers=AUTH).status_code == 404


def test_status_and_result_lifecycle(client):
    job_id = str(uuid.uuid4())
    jobs.register_job(job_id, f'{job_id}.wav', 'x.wav')
    status = lambda: client.get(f'/transcriptions/{job_id}', headers=AUTH)
    result = lambda: client.get(f'/transcriptions/{job_id}/result', headers=AUTH)

    assert status().json() == {'job_id': job_id, 'status': 'pending'}
    assert result().status_code == 202

    FakeAsyncResult.states[job_id] = ('PROGRESS', {'completed_chunks': 1, 'total_chunks': 3})
    assert status().json()['progress'] == {'completed_chunks': 1, 'total_chunks': 3}

    payload = {'filename': 'x.wav', 'language': 'en', 'transcription': 'hi',
               'segments': [{'start': 0.0, 'end': 1.0, 'text': 'hi'}]}
    FakeAsyncResult.states[job_id] = ('SUCCESS', payload)
    assert status().json()['result_url'] == f'/transcriptions/{job_id}/result'
    assert result().status_code == 200 and result().json() == payload


def test_failed_job_reports_recorded_error(client):
    job_id = str(uuid.uuid4())
    jobs.register_job(job_id, f'{job_id}.wav', 'x.wav')
    jobs.mark_failed(job_id, 'Audio could not be decoded')
    FakeAsyncResult.states[job_id] = ('FAILURE', None)
    assert client.get(f'/transcriptions/{job_id}', headers=AUTH).json()['error'] == 'Audio could not be decoded'
    response = client.get(f'/transcriptions/{job_id}/result', headers=AUTH)
    assert response.status_code == 422 and response.json()['detail'] == 'Audio could not be decoded'


def test_redis_outage_returns_503_and_removes_upload(client, queued, probe_ok, monkeypatch):
    unreachable = redis.Redis(host='127.0.0.1', port=1, socket_connect_timeout=1, decode_responses=True)
    monkeypatch.setattr(jobs, 'redis_client', unreachable)
    response = upload(client)
    assert response.status_code == 503
    assert queued == [] and uploads_on_disk() == []
    assert client.get(f'/transcriptions/{uuid.uuid4()}', headers=AUTH).status_code == 503


def test_broker_outage_returns_503_and_unregisters_job(client, probe_ok, monkeypatch):
    from kombu.exceptions import OperationalError

    def broker_down(args, task_id):
        raise OperationalError('broker unreachable')
    monkeypatch.setattr(main.transcribe_audio, 'apply_async', broker_down)
    response = upload(client)
    assert response.status_code == 503
    assert uploads_on_disk() == []
    assert jobs.redis_client.keys('transcription:job:*') == []


def test_api_refuses_to_start_without_api_key(monkeypatch):
    monkeypatch.setattr(main, 'API_KEY', '')
    monkeypatch.setattr(main, 'check_binaries', lambda *names: None)
    with pytest.raises(RuntimeError, match='API_KEY'):
        with TestClient(main.app):
            pass
