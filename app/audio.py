"""FFmpeg/FFprobe helpers shared by the API and the worker."""
import json
import subprocess
from pathlib import Path

from app.config import (
    FFMPEG_TIMEOUT_SECONDS,
    FFPROBE_TIMEOUT_SECONDS,
    MAX_AUDIO_SECONDS,
)


class InvalidAudioError(ValueError):
    """The input cannot be processed. Retrying will not help.

    The message is safe to show to API clients.
    """


class AudioTooLongError(InvalidAudioError):
    """The audio is decodable but longer than MAX_AUDIO_SECONDS."""


def check_binaries(*names: str) -> None:
    """Fail fast if a required binary is missing or cannot execute.

    `shutil.which` is not enough: on Windows an application-control policy can
    block an executable that is on PATH.
    """
    for name in names:
        try:
            subprocess.run([name, '-version'], capture_output=True, check=True, timeout=15)
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError(f'{name} is required on PATH and must be executable: {exc}') from exc


def probe_audio(path: Path) -> float:
    """Return the duration in seconds of a decodable audio file.

    Raises InvalidAudioError for files that are not audio, cannot be parsed, have
    no usable duration, or exceed MAX_AUDIO_SECONDS.
    """
    try:
        completed = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration:stream=codec_type',
             '-of', 'json', str(path)],
            capture_output=True, text=True, check=True, timeout=FFPROBE_TIMEOUT_SECONDS,
        )
        info = json.loads(completed.stdout or '{}')
    except subprocess.TimeoutExpired:
        raise InvalidAudioError('Audio file could not be analysed in time') from None
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        raise InvalidAudioError('File is not a decodable audio file') from None

    if not any(stream.get('codec_type') == 'audio' for stream in info.get('streams', [])):
        raise InvalidAudioError('File contains no audio stream')
    try:
        duration = float(info.get('format', {}).get('duration'))
    except (TypeError, ValueError):
        raise InvalidAudioError('Audio duration could not be determined') from None
    if not duration > 0:  # also rejects NaN
        raise InvalidAudioError('Audio duration must be positive')
    if duration > MAX_AUDIO_SECONDS:
        raise AudioTooLongError(
            f'Audio is {duration:.0f}s long; the limit is {MAX_AUDIO_SECONDS}s')
    return duration


def convert_chunk(source: Path, output: Path, start: float, duration: float) -> None:
    """Decode part of the source to 16 kHz mono WAV, the format Whisper expects."""
    try:
        subprocess.run(
            ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
             '-ss', f'{start:.3f}', '-i', str(source), '-t', f'{duration:.3f}',
             '-vn', '-ac', '1', '-ar', '16000', str(output)],
            capture_output=True, check=True, timeout=FFMPEG_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        raise InvalidAudioError('Audio decoding timed out') from None
    except subprocess.CalledProcessError:
        raise InvalidAudioError('Audio could not be decoded') from None
