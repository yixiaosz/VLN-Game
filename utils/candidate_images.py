"""Request-specific candidate image publication and single-process GUI handoff."""
from concurrent.futures import Future, TimeoutError as FutureTimeout
from dataclasses import dataclass
import hashlib
from io import BytesIO
import os
from pathlib import Path
import re
import tempfile
import threading

from PIL import Image


PROTOCOL = 'request_scoped_atomic_completion_v1'
RENDER_TIMEOUT_SECONDS = 120.0


class CandidateRenderError(RuntimeError):
    pass


class CandidateRenderCancelled(CandidateRenderError):
    pass


@dataclass(frozen=True)
class CandidateRenderRequest:
    request_id: str
    run_id: str
    episode_number: int
    search_id: int
    candidate_id: int
    path: str
    first_view_sha256: str
    first_view_shape: tuple
    first_view_dtype: str


def make_request(dump_location, experiment, run_id, episode_number, search_id,
                 candidate_id, first_view):
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', run_id):
        raise CandidateRenderError('Invalid render run identity')
    episode_number, search_id, candidate_id = map(int, (episode_number, search_id, candidate_id))
    path = (Path(dump_location) / experiment / 'episodes' / str(episode_number) /
            'candidates' / run_id / f'search-{search_id:03d}' / f'candidate-{candidate_id}.jpeg')
    request_id = f'{run_id}:episode-{episode_number}:search-{search_id}:candidate-{candidate_id}'
    return CandidateRenderRequest(
        request_id, run_id, episode_number, search_id, candidate_id, str(path.resolve()),
        hashlib.sha256(first_view.tobytes()).hexdigest(), tuple(first_view.shape), str(first_view.dtype))


def validate_first_view(request, image):
    if (tuple(image.shape) != request.first_view_shape or str(image.dtype) != request.first_view_dtype
            or hashlib.sha256(image.tobytes()).hexdigest() != request.first_view_sha256):
        raise CandidateRenderError(f'First-view image does not match render request {request.request_id}')


def validate_completion(request, completion):
    expected = {
        'request_id': request.request_id, 'run_id': request.run_id,
        'episode_number': request.episode_number, 'search_id': request.search_id,
        'candidate_id': request.candidate_id, 'path': request.path,
        'first_view_sha256': request.first_view_sha256,
        'first_view_shape': list(request.first_view_shape),
        'first_view_dtype': request.first_view_dtype, 'dimensions': [960, 480],
        'protocol': PROTOCOL,
    }
    for name, value in expected.items():
        if completion.get(name) != value:
            raise CandidateRenderError(f'Render completion {name} mismatch for {request.request_id}')
    if not re.fullmatch(r'[0-9a-f]{64}', completion.get('sha256', '')):
        raise CandidateRenderError(f'Missing/invalid image hash for {request.request_id}')


def publish_candidate(request, image):
    """Publish the original composite as JPEG, then acknowledge its exact bytes."""
    if image.size != (960, 480):
        raise CandidateRenderError(f'Candidate composite must be 960x480, got {image.size}')
    buffer = BytesIO()
    image.save(buffer, format='JPEG')
    data = buffer.getvalue()
    path = Path(request.path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.candidate-',
                                         suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return {
        'protocol': PROTOCOL, 'request_id': request.request_id, 'run_id': request.run_id,
        'episode_number': request.episode_number, 'search_id': request.search_id,
        'candidate_id': request.candidate_id, 'path': request.path,
        'sha256': hashlib.sha256(data).hexdigest(), 'dimensions': [960, 480],
        'first_view_sha256': request.first_view_sha256,
        'first_view_shape': list(request.first_view_shape),
        'first_view_dtype': request.first_view_dtype,
    }


def read_completed_image(request, completion):
    """Read once, only after matching completion; never poll arbitrary Vis files."""
    validate_completion(request, completion)
    try:
        data = Path(request.path).read_bytes()
        if hashlib.sha256(data).hexdigest() != completion['sha256']:
            raise CandidateRenderError(f'Completed image bytes changed for {request.request_id}')
        with Image.open(BytesIO(data)) as decoded:
            if decoded.size != (960, 480) or decoded.format != 'JPEG':
                raise CandidateRenderError(f'Invalid completed candidate format for {request.request_id}')
            decoded.load()
    except (OSError, ValueError) as exc:
        raise CandidateRenderError(f'Cannot read completed candidate {request.request_id}: {exc}') from exc
    return data


class CandidateImageRequests:
    """Shared by navigation, UpdateMain, and GUI callbacks in one process.

    Only worker/navigation threads wait. GUI callbacks publish a result or an
    error and return; closing the GUI wakes all pending waiters immediately.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._requests = {}
        self._closed = False
        self._gui_close = None
        self._gui_close_requested = False

    def check_open(self):
        with self._lock:
            if self._closed:
                raise CandidateRenderCancelled('Candidate renderer is closed')

    def register(self, request):
        with self._lock:
            if self._closed:
                raise CandidateRenderCancelled('Candidate renderer is closed')
            if request.request_id in self._requests:
                raise CandidateRenderError(f'Duplicate render request {request.request_id}')
            self._requests[request.request_id] = (request, Future())

    def _future(self, request):
        try:
            stored, future = self._requests[request.request_id]
        except KeyError as exc:
            raise CandidateRenderError(f'Unregistered render request {request.request_id}') from exc
        if stored != request:
            raise CandidateRenderError(f'Render request identity mismatch: {request.request_id}')
        return future

    def complete(self, request, completion):
        validate_completion(request, completion)
        with self._lock:
            future = self._future(request)
            if self._closed or future.done():
                return False
            future.set_result(dict(completion))
            return True

    def fail(self, request, error):
        with self._lock:
            future = self._future(request)
            if future.done():
                return False
            future.set_exception(CandidateRenderError(f'{request.request_id}: {error}'))
            return True

    def wait(self, request, timeout=RENDER_TIMEOUT_SECONDS):
        with self._lock:
            if self._closed:
                raise CandidateRenderCancelled('Candidate renderer is closed')
            future = self._future(request)
        try:
            return future.result(timeout=timeout)
        except FutureTimeout:
            self.fail(request, f'Render completion timed out after {timeout:g}s')
            return future.result()

    def set_gui_close_callback(self, callback):
        with self._lock:
            self._gui_close = callback

    def request_gui_close(self):
        with self._lock:
            if self._closed or self._gui_close_requested:
                return
            callback = self._gui_close
            if callback is not None:
                self._gui_close_requested = True
        if callback is not None:
            callback()

    def close(self):
        with self._lock:
            self._closed = True
            for request, future in self._requests.values():
                if not future.done():
                    future.set_exception(CandidateRenderCancelled(
                        f'GUI closed before render completed: {request.request_id}'))
