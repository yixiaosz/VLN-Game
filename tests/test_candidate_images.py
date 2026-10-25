"""Offline image handoff/race tests; GUI methods are exercised without a window."""
from contextlib import redirect_stdout
import hashlib
from io import BytesIO, StringIO
import json
import logging
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from utils.candidate_images import (
    CandidateImageRequests, CandidateRenderError, CandidateRenderCancelled,
    make_request, publish_candidate, read_completed_image, validate_first_view)
from utils.vis_gui import ReconstructionWindow
from utils.equ_trace import EquilibriumTraceRun


class CandidateImageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='candidate-image-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.channel = CandidateImageRequests()
        self.addCleanup(self.channel.close)
        self.first_view = np.full((480, 640, 3), (30, 100, 190), dtype=np.uint8)
        self.top = np.full((800, 640, 3), (160, 60, 20), dtype=np.uint8)
        self.window = ReconstructionWindow.__new__(ReconstructionWindow)
        self.window.args = SimpleNamespace(dump_location=str(self.root), exp_name='vlobjectnav-vlm_game',
                                           path_npz=str(self.root) + '/')
        self.window.candidate_images = self.channel
        self.window.is_done = False
        self.window.candidate_id = 0
        self.window.saved_objects = None
        self.window.saved_full_points = None
        self.window.saved_full_colors = None

    def request(self, run='run-A', episode=1, search=1, candidate=15, first_view=None):
        request = make_request(self.root, self.window.args.exp_name, run, episode, search,
                               candidate, self.first_view if first_view is None else first_view)
        self.channel.register(request)
        return request

    def render(self, request, first_view=None):
        self.window.on_image_rendered(self.top, request.episode_number, request.candidate_id,
                                      self.first_view if first_view is None else first_view, request)
        completion = self.channel.wait(request, timeout=1)
        return completion, read_completed_image(request, completion)

    def test_existing_vis_and_correct_size_stale_file_cannot_signal_readiness(self):
        request = self.request()
        legacy = self.root / self.window.args.exp_name / 'episodes' / '1' / 'Vis-15.jpeg'
        legacy.parent.mkdir(parents=True)
        legacy.write_bytes(b'old full-scene image')
        # Even a valid 960x480 file at the new path must not count as completion.
        path = Path(request.path)
        path.parent.mkdir(parents=True)
        Image.new('RGB', (960, 480), 'black').save(path)
        done = threading.Event()
        result = []
        def consumer():
            result.append(self.channel.wait(request, timeout=2))
            done.set()
        thread = threading.Thread(target=consumer)
        thread.start()
        self.assertFalse(done.wait(0.05))
        completion, data = self.render(request)
        self.assertTrue(done.wait(1))
        thread.join(1)
        self.assertEqual(result[0], completion)
        self.assertEqual(legacy.read_bytes(), b'old full-scene image')
        with Image.open(BytesIO(data)) as image:
            self.assertEqual(image.size, (960, 480))
            actual = np.asarray(image)[100:400, 600:900].mean(axis=(0, 1))
        np.testing.assert_allclose(actual, self.first_view[0, 0], atol=4)

    def test_late_previous_request_cannot_complete_next_search(self):
        old = self.request(search=1)
        newer_view = np.full_like(self.first_view, (190, 40, 80))
        new = self.request(search=2, first_view=newer_view)
        done = threading.Event()
        results = []
        def consumer():
            results.append(self.channel.wait(new, timeout=2))
            done.set()
        thread = threading.Thread(target=consumer)
        thread.start()
        self.render(old)
        self.assertFalse(done.wait(0.05))
        completion, data = self.render(new, newer_view)
        self.assertTrue(done.wait(1))
        thread.join(1)
        self.assertEqual(results[0]['request_id'], new.request_id)
        self.assertNotEqual(old.path, new.path)
        self.assertNotEqual(old.first_view_sha256, new.first_view_sha256)
        with Image.open(BytesIO(data)) as image:
            actual = np.asarray(image)[100:400, 600:900].mean(axis=(0, 1))
        np.testing.assert_allclose(actual, newer_view[0, 0], atol=4)

    def test_run_episode_search_and_candidate_are_all_part_of_identity(self):
        requests = [self.request(), self.request(run='run-B'), self.request(episode=2),
                    self.request(search=2), self.request(candidate=24)]
        self.assertEqual(len({request.path for request in requests}), 5)
        self.assertEqual(len({request.request_id for request in requests}), 5)
        with self.assertRaisesRegex(CandidateRenderError, 'Duplicate'):
            self.channel.register(requests[0])

    def test_ordinary_frames_use_a_separate_namespace(self):
        request = self.request()
        completion, data = self.render(request)
        self.window.on_top_rendered(self.top, 1, 15)
        frame = self.root / self.window.args.exp_name / 'episodes' / '1' / 'frames' / 'Frame-15.jpeg'
        self.assertTrue(frame.is_file())
        self.assertEqual(Path(request.path).read_bytes(), data)
        self.assertEqual(read_completed_image(request, completion), data)
        with Image.open(frame) as image:
            self.assertEqual(image.size, (640, 800))

    def test_publication_is_atomic_and_does_not_ack_partial_bytes(self):
        request = self.request()
        original_replace = __import__('os').replace
        def before_publish(source, destination):
            self.assertFalse(Path(destination).exists())
            with Image.open(source) as image:
                image.load()
                self.assertEqual(image.size, (960, 480))
            self.assertFalse(self.channel._requests[request.request_id][1].done())
            return original_replace(source, destination)
        with patch('utils.candidate_images.os.replace', side_effect=before_publish):
            completion, data = self.render(request)
        self.assertEqual(completion['sha256'], hashlib.sha256(data).hexdigest())
        self.assertFalse(list(Path(request.path).parent.glob('.candidate-*.tmp')))

    def test_incorrect_first_view_is_rejected_before_render_and_callback(self):
        request = self.request()
        wrong = np.zeros_like(self.first_view)
        with self.assertRaises(CandidateRenderError):
            validate_first_view(request, wrong)
        self.window._render_top_view_map = Mock()
        with redirect_stdout(StringIO()):
            self.window.render_top_view_map([{}] * 16, [], [], 15, wrong, 1, request)
        self.window._render_top_view_map.assert_not_called()
        with self.assertRaisesRegex(CandidateRenderError, 'First-view'):
            self.channel.wait(request, timeout=0.1)
        self.assertFalse(Path(request.path).exists())

    def test_packet_identity_error_and_native_render_exception_are_reported(self):
        mismatched = self.request()
        with redirect_stdout(StringIO()):
            self.window.render_top_view_map([{}] * 16, [], [], 15, self.first_view, 2, mismatched)
        with self.assertRaisesRegex(CandidateRenderError, 'identity mismatch'):
            self.channel.wait(mismatched, timeout=0.1)
        failed = self.request(search=2)
        self.window._render_top_view_map = Mock(side_effect=RuntimeError('synthetic render failure'))
        with redirect_stdout(StringIO()):
            self.window.render_top_view_map([{}] * 16, [], [], 15, self.first_view, 1, failed)
        with self.assertRaisesRegex(CandidateRenderError, 'synthetic render failure'):
            self.channel.wait(failed, timeout=0.1)

    def test_publication_error_wakes_waiter_without_accepting_a_file(self):
        request = self.request()
        with patch('utils.vis_gui.publish_candidate', side_effect=OSError('synthetic disk error')):
            with redirect_stdout(StringIO()):
                self.window.on_image_rendered(self.top, 1, 15, self.first_view, request)
        with self.assertRaisesRegex(CandidateRenderError, 'synthetic disk error'):
            self.channel.wait(request, timeout=0.1)
        self.assertFalse(Path(request.path).exists())

    def test_gui_close_cancels_waiters_and_late_callbacks(self):
        request = self.request()
        done = threading.Event()
        failures = []
        def consumer():
            try:
                self.channel.wait(request, timeout=5)
            except CandidateRenderCancelled as exc:
                failures.append(str(exc))
            finally:
                done.set()
        thread = threading.Thread(target=consumer)
        thread.start()
        with redirect_stdout(StringIO()):
            self.window._on_close()
        self.assertTrue(done.wait(1))
        thread.join(1)
        self.assertEqual(len(failures), 1)
        with redirect_stdout(StringIO()):
            self.window.on_image_rendered(self.top, 1, 15, self.first_view, request)
        self.assertFalse(Path(request.path).exists())
        with self.assertRaises(CandidateRenderCancelled):
            self.request(search=2)

    def test_timeout_is_bounded_and_late_completion_is_not_accepted(self):
        request = self.request()
        with self.assertRaisesRegex(CandidateRenderError, 'timed out'):
            self.channel.wait(request, timeout=0.01)
        completion = publish_candidate(request, Image.new('RGB', (960, 480)))
        self.assertFalse(self.channel.complete(request, completion))
        with self.assertRaisesRegex(CandidateRenderError, 'timed out'):
            self.channel.wait(request, timeout=0.01)

    def test_error_shutdown_posts_gui_close_only_once(self):
        close = Mock()
        self.channel.set_gui_close_callback(close)
        self.channel.request_gui_close()
        self.channel.request_gui_close()
        close.assert_called_once_with()
        self.channel.close()
        self.channel.request_gui_close()
        close.assert_called_once_with()

    def test_wrong_dimensions_metadata_and_changed_bytes_are_rejected(self):
        request = self.request()
        with self.assertRaisesRegex(CandidateRenderError, '960x480'):
            publish_candidate(request, Image.new('RGB', (640, 800)))
        completion = publish_candidate(request, Image.new('RGB', (960, 480)))
        for field, value in [('candidate_id', 24), ('search_id', 2), ('dimensions', [640, 800]),
                             ('path', 'Vis-15.jpeg'), ('first_view_sha256', 'wrong')]:
            wrong = dict(completion, **{field: value})
            with self.assertRaises(CandidateRenderError):
                read_completed_image(request, wrong)
        Path(request.path).write_bytes(b'partial JPEG')
        with self.assertRaisesRegex(CandidateRenderError, 'bytes changed'):
            read_completed_image(request, completion)

    def test_archive_links_completion_to_exact_api_buffer(self):
        request = self.request()
        completion, data = self.render(request)
        logger = logging.getLogger('candidate-image-trace-test')
        logger.addHandler(logging.NullHandler())
        run = EquilibriumTraceRun(self.root, {}, {}, logger)
        episode = run.start_episode(1, '0', 'scene', 'instruction')
        search = episode.start_search(80, [15])
        search.archive_image(15, 0, request.path, data, render_completion=completion)
        episode.finish('completed', 100)
        events = [json.loads(line) for line in episode.path.read_text().splitlines()]
        image = next(e for e in events if e['event'] == 'image_input')
        self.assertEqual(image['sha256'], image['render_completion']['sha256'])
        self.assertEqual(image['render_completion']['request_id'], request.request_id)
        self.assertEqual((run.root / image['archive_path']).read_bytes(), data)


if __name__ == '__main__':
    unittest.main()
