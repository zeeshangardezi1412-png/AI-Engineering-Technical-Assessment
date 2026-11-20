import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

from app import audio, jobs, tasks
from app.audio import AudioTooLongError, InvalidAudioError
from app.config import UPLOAD_DIR
from conftest import FakeWhisper, make_wav, requires_ffprobe


@pytest.fixture
def model(monkeypatch):
    fake = FakeWhisper()
    monkeypatch.setattr(tasks, 'get_model', lambda: fake)
    monkeypatch.setattr(tasks, 'convert_chunk', lambda src, out, start, duration: fake.record_window(start, duration))
    monkeypatch.setattr(tasks.transcribe_audio, 'update_state', lambda **kwargs: None)
    return fake


def new_job(duration=150.0, monkeypatch=None):
    job_id = str(uuid.uuid4())
    stored_name = f'{job_id}.wav'
    (Path(UPLOAD_DIR) / stored_name).write_bytes(b'audio')
    jobs.register_job(job_id, stored_name, 'talk.wav')
    if monkeypatch is not None:
        monkeypatch.setattr(tasks, 'probe_audio', lambda path: duration)
    return job_id, stored_name


def run(job_id, stored_name):
    return tasks.transcribe_audio.apply(args=[stored_name, 'talk.wav'], task_id=job_id)


# --- chunk boundaries ------------------------------------------------------------

def test_window_drops_cut_off_last_segment_and_resumes_at_its_start():
    raw = [{'start': 0, 'end': 20, 'text': 'a'}, {'start': 20, 'end': 52, 'text': 'b'},
           {'start': 52, 'end': 60, 'text': 'cut off'}]
    segments, next_start = tasks._window_segments(raw, 100.0, 160.0, is_final=False)
    assert [s['text'] for s in segments] == ['a', 'b']
    assert segments[0] == {'start': 100.0, 'end': 120.0, 'text': 'a'}
    assert next_start == 152.0


def test_final_window_keeps_every_segment():
    raw = [{'start': 0, 'end': 20, 'text': 'a'}, {'start': 40, 'end': 45, 'text': 'end'}]
    segments, next_start = tasks._window_segments(raw, 0.0, 45.0, is_final=True)
    assert [s['text'] for s in segments] == ['a', 'end'] and next_start == 45.0


@pytest.mark.parametrize('raw', [
    [],                                                     # silence
    [{'start': 0, 'end': 60, 'text': 'one long segment'}],  # nothing to drop
    [{'start': 0, 'end': 10, 'text': 'a'}, {'start': 10, 'end': 60, 'text': 'starts too early'}],
])
def test_window_falls_back_to_hard_cut_so_progress_is_guaranteed(raw):
    segments, next_start = tasks._window_segments(raw, 0.0, 60.0, is_final=False)
    assert next_start == 60.0 and len(segments) == len(raw)


def test_segment_times_are_clamped_to_window_and_blank_text_dropped():
    raw = [{'start': 1, 'end': 99, 'text': ' hi '}, {'start': 2, 'end': 3, 'text': '   '}]
    segments, _ = tasks._window_segments(raw, 10.0, 40.0, is_final=True)
    assert segments == [{'start': 11.0, 'end': 40.0, 'text': 'hi'}]


# --- end-to-end task behaviour with a fake model ------------------------------------

def test_successful_job_returns_timestamped_json(model, monkeypatch):
    job_id, stored_name = new_job(150.0, monkeypatch)
    result = run(job_id, stored_name)
    assert result.state == 'SUCCESS'
    body = result.result
    assert body['filename'] == 'talk.wav' and body['language'] == 'en'
    starts = [s['start'] for s in body['segments']]
    assert starts == sorted(starts) and starts[0] == 0.0 and body['segments'][-1]['end'] == 150.0
    assert all(s['end'] >= s['start'] for s in body['segments'])
    assert body['transcription'] == ' '.join(s['text'] for s in body['segments'])
    # Each window after the first starts where the previous window's cut-off segment began.
    assert model.windows == [(0.0, 60.0), (50.0, 60.0), (100.0, 50.0)]
    texts = [s['text'] for s in body['segments']]
    assert texts == [f't{t}' for t in range(0, 150, 10)]  # every 10 s exactly once
    # Upload, checkpoint and lock are gone; job is marked completed.
    assert not (Path(UPLOAD_DIR) / stored_name).exists()
    assert jobs.redis_client.keys('transcription:progress:*') == []
    assert jobs.redis_client.keys('transcription:lock:*') == []
    assert 'completed_at' in jobs.get_job(job_id)


def test_invalid_audio_fails_once_without_retry(model, monkeypatch):
    job_id, stored_name = new_job()
    probes = []

    def bad_probe(path):
        probes.append(path)
        raise InvalidAudioError('File is not a decodable audio file')
    monkeypatch.setattr(tasks, 'probe_audio', bad_probe)
    result = run(job_id, stored_name)
    assert result.state == 'FAILURE'
    assert len(probes) == 1  # no retries for a permanent error
    assert jobs.get_job(job_id)['error'] == 'File is not a decodable audio file'
    assert not (Path(UPLOAD_DIR) / stored_name).exists()


def test_decode_failure_mid_job_fails_without_retry(model, monkeypatch):
    job_id, stored_name = new_job(150.0, monkeypatch)

    def convert(src, out, start, duration):
        model.record_window(start, duration)
        if start > 0:
            raise InvalidAudioError('Audio could not be decoded')
    monkeypatch.setattr(tasks, 'convert_chunk', convert)
    assert run(job_id, stored_name).state == 'FAILURE'
    assert len(model.windows) == 2
    assert jobs.redis_client.keys('transcription:progress:*') == []
    assert not (Path(UPLOAD_DIR) / stored_name).exists()


def test_transient_error_retries_and_resumes_from_checkpoint(model, monkeypatch):
    job_id, stored_name = new_job(150.0, monkeypatch)
    model.fail_on_call[1] = RuntimeError('worker ran out of memory')  # second chunk, first try
    result = run(job_id, stored_name)
    assert result.state == 'SUCCESS'
    # Chunk 0 is not transcribed again after the retry.
    assert [start for start, _ in model.windows] == [0.0, 50.0, 50.0, 100.0]
    texts = [s['text'] for s in result.result['segments']]
    assert len(texts) == len(set(texts))


def test_retries_are_bounded_then_job_fails_and_cleans_up(model, monkeypatch):
    job_id, stored_name = new_job(30.0, monkeypatch)
    for call in range(10):
        model.fail_on_call[call] = RuntimeError('model crashed')
    result = run(job_id, stored_name)
    assert result.state == 'FAILURE'
    assert len(model.windows) == tasks.MAX_RETRIES + 1
    assert 'after retries' in jobs.get_job(job_id)['error']
    assert not (Path(UPLOAD_DIR) / stored_name).exists()
    assert jobs.redis_client.keys('transcription:lock:*') == []


def test_missing_source_file_fails_without_retry(model, monkeypatch):
    job_id, stored_name = new_job(30.0, monkeypatch)
    (Path(UPLOAD_DIR) / stored_name).unlink()
    assert run(job_id, stored_name).state == 'FAILURE'
    assert jobs.get_job(job_id)['error'] == 'Uploaded file is no longer available'
    assert model.windows == []


def test_duplicate_execution_respects_lock(model, monkeypatch):
    job_id, stored_name = new_job(30.0, monkeypatch)
    other_worker = jobs.redis_client.lock(jobs.lock_key(job_id), timeout=60)
    assert other_worker.acquire(blocking=False)
    result = run(job_id, stored_name)
    assert result.state == 'IGNORED'
    assert model.windows == []                       # no work done
    assert (Path(UPLOAD_DIR) / stored_name).exists()  # owner's file untouched
    assert other_worker.owned()                       # owner's lock untouched
    assert not jobs.is_terminal(jobs.get_job(job_id))


def test_worker_that_loses_its_lock_stops_without_writing(model, monkeypatch):
    job_id, stored_name = new_job(150.0, monkeypatch)
    original = model.transcribe

    def steal_lock_then_transcribe(*args, **kwargs):
        jobs.redis_client.set(jobs.lock_key(job_id), 'other-worker-token')
        return original(*args, **kwargs)
    model.transcribe = steal_lock_then_transcribe
    assert run(job_id, stored_name).state == 'IGNORED'
    assert jobs.redis_client.llen(jobs.segments_key(job_id)) == 0
    assert jobs.redis_client.get(jobs.lock_key(job_id)) == 'other-worker-token'
    assert (Path(UPLOAD_DIR) / stored_name).exists()


def test_duplicate_delivery_of_finished_job_is_ignored(model, monkeypatch):
    job_id, stored_name = new_job(30.0, monkeypatch)
    assert run(job_id, stored_name).state == 'SUCCESS'
    assert run(job_id, stored_name).state == 'IGNORED'
    assert len(model.windows) == 1


def test_unregistered_job_is_dropped_and_file_removed(model, monkeypatch):
    job_id, stored_name = new_job(30.0, monkeypatch)
    jobs.delete_job(job_id)
    assert run(job_id, stored_name).state == 'IGNORED'
    assert not (Path(UPLOAD_DIR) / stored_name).exists()
    assert model.windows == []


def test_checkpoint_from_different_chunk_plan_is_discarded(model, monkeypatch):
    job_id, stored_name = new_job(150.0, monkeypatch)
    stale_plan = {'chunk_seconds': 30, 'duration': 150.0}
    jobs.save_chunk(job_id, stale_plan, [{'start': 0, 'end': 1, 'text': 'stale'}], 1, 30.0, 'fr')
    result = run(job_id, stored_name)
    assert result.state == 'SUCCESS'
    assert 'stale' not in result.result['transcription']
    assert model.windows[0] == (0.0, 60.0)


@pytest.mark.parametrize('stored_name', ['../../etc/passwd', '..\\secret.wav', '/etc/passwd', 'sub/x.wav'])
def test_task_rejects_paths_outside_upload_dir(model, stored_name):
    job_id = str(uuid.uuid4())
    jobs.register_job(job_id, stored_name, 'x.wav')
    assert run(job_id, stored_name).state == 'FAILURE'
    assert jobs.get_job(job_id)['error'] == 'Invalid upload reference'


# --- cleanup ------------------------------------------------------------------------

def test_orphan_cleanup_removes_only_stale_unowned_files():
    upload_dir = Path(UPLOAD_DIR)
    old = time.time() - 2 * 3600
    active_id, done_id, orphan_id, fresh_id = (str(uuid.uuid4()) for _ in range(4))
    jobs.register_job(active_id, f'{active_id}.wav', 'a.wav')
    jobs.register_job(done_id, f'{done_id}.wav', 'b.wav')
    jobs.mark_completed(done_id)
    for job_id in (active_id, done_id, orphan_id, fresh_id):
        (upload_dir / f'{job_id}.wav').write_bytes(b'x')
    (upload_dir / 'junk.bin').write_bytes(b'x')
    for name in (f'{active_id}.wav', f'{done_id}.wav', f'{orphan_id}.wav', 'junk.bin'):
        os.utime(upload_dir / name, (old, old))

    assert tasks.cleanup_orphaned_uploads() == 3
    assert sorted(p.name for p in upload_dir.iterdir()) == sorted([f'{active_id}.wav', f'{fresh_id}.wav'])


# --- ffprobe / ffmpeg wrappers ------------------------------------------------------

def fake_ffprobe(monkeypatch, payload):
    monkeypatch.setattr(audio.subprocess, 'run', lambda *a, **k: subprocess.CompletedProcess(a, 0, json.dumps(payload), ''))


@pytest.mark.parametrize('payload, message', [
    ({'streams': [{'codec_type': 'video'}], 'format': {'duration': '5'}}, 'no audio stream'),
    ({'streams': [{'codec_type': 'audio'}], 'format': {}}, 'could not be determined'),
    ({'streams': [{'codec_type': 'audio'}], 'format': {'duration': 'N/A'}}, 'could not be determined'),
    ({'streams': [{'codec_type': 'audio'}], 'format': {'duration': '0'}}, 'must be positive'),
    ({'streams': [{'codec_type': 'audio'}], 'format': {'duration': 'nan'}}, 'must be positive'),
])
def test_probe_rejects_unusable_metadata(monkeypatch, payload, message):
    fake_ffprobe(monkeypatch, payload)
    with pytest.raises(InvalidAudioError, match=message):
        audio.probe_audio(Path('x.wav'))


def test_probe_rejects_audio_over_duration_limit(monkeypatch):
    fake_ffprobe(monkeypatch, {'streams': [{'codec_type': 'audio'}], 'format': {'duration': '601'}})
    with pytest.raises(AudioTooLongError):
        audio.probe_audio(Path('x.wav'))


def test_hung_ffprobe_is_killed_by_timeout(monkeypatch):
    real_run = subprocess.run
    monkeypatch.setattr(audio, 'FFPROBE_TIMEOUT_SECONDS', 1)
    # Run a process that never finishes in place of ffprobe; the real timeout applies.
    monkeypatch.setattr(audio.subprocess, 'run',
                        lambda cmd, **kw: real_run([sys.executable, '-c', 'import time; time.sleep(60)'], **kw))
    started = time.monotonic()
    with pytest.raises(InvalidAudioError, match='in time'):
        audio.probe_audio(Path('x.wav'))
    assert time.monotonic() - started < 10


def test_missing_binary_fails_fast():
    with pytest.raises(RuntimeError, match='required on PATH'):
        audio.check_binaries('definitely-not-a-real-binary-xyz')


@requires_ffprobe
def test_real_ffprobe_reads_wav_duration(tmp_path):
    path = tmp_path / 'tone.wav'
    path.write_bytes(make_wav(2.0))
    assert audio.probe_audio(path) == pytest.approx(2.0, abs=0.05)
