"""Offscreen handoff/failure tests; actual EGL rendering is tested separately."""
import ast
from io import BytesIO
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from utils.candidate_images import CandidateRenderError, make_request, read_completed_image
from utils.candidate_renderer import CandidateRenderer
from utils import candidate_renderer as module


class RendererTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.native = Mock()
        self.native.render_to_image.return_value = np.full((800, 640, 3), 200, dtype=np.uint8)
        factory = patch.object(module.rendering, 'OffscreenRenderer', return_value=self.native)
        factory.start()
        self.addCleanup(factory.stop)
        self.renderer = CandidateRenderer()
        self.addCleanup(self.renderer.close)
        self.first = np.full((480, 640, 3), [30, 100, 190], dtype=np.uint8)
        self.points = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)

    def render(self, episode=1, search=1, candidate=15):
        request = make_request(self.directory.name, 'test', 'run', episode, search, candidate, self.first)
        completion = self.renderer.render(request, self.points, self.points,
                                          [[0, 0, 0], [1, 1, 1]], self.first)
        return request, completion, read_completed_image(request, completion)

    def test_repeated_candidate_identity_and_first_view_are_preserved(self):
        records = [self.render(), self.render(search=2), self.render(episode=2)]
        self.assertEqual(len({r[0].path for r in records}), 3)
        for request, completion, data in records:
            self.assertEqual(completion['first_view_sha256'], request.first_view_sha256)
            with Image.open(BytesIO(data)) as image:
                self.assertEqual(image.size, (960, 480))
                np.testing.assert_allclose(np.asarray(image)[100, 700], self.first[0, 0], atol=3)

    def test_bad_first_view_never_renders_or_publishes(self):
        request = make_request(self.directory.name, 'test', 'run', 1, 1, 0, self.first)
        with self.assertRaises(CandidateRenderError):
            self.renderer.render(request, self.points, self.points, [[0]*3, [1]*3], self.first + 1)
        self.native.render_to_image.assert_not_called()
        self.assertFalse(Path(request.path).exists())

    def test_native_python_error_is_explicit_without_publishing(self):
        self.native.render_to_image.side_effect = RuntimeError('synthetic EGL render failure')
        with self.assertRaisesRegex(CandidateRenderError, 'synthetic EGL'):
            self.render()
        self.assertFalse(list(Path(self.directory.name).rglob('*.jpeg')))

    def test_publication_error_is_propagated(self):
        with patch.object(module, 'publish_candidate', side_effect=OSError('synthetic disk error')):
            with self.assertRaisesRegex(CandidateRenderError, 'synthetic disk error'):
                self.render()

    def test_renderer_rejects_cross_thread_use(self):
        errors = []
        def call():
            try:
                self.render()
            except CandidateRenderError as exc:
                errors.append(str(exc))
        thread = threading.Thread(target=call)
        thread.start()
        thread.join(2)
        self.assertEqual(len(errors), 1)
        self.assertIn('creating thread', errors[0])
        self.native.render_to_image.assert_not_called()

    def test_close_is_idempotent_and_prevents_later_render(self):
        self.renderer.close()
        self.renderer.close()
        with self.assertRaisesRegex(CandidateRenderError, 'closed'):
            self.render()

    def test_entrypoint_headless_failure_closes_renderer_and_returns_nonzero(self):
        # Run the real __main__ branch with fake main/renderer; no Habitat/API.
        source = ast.parse(Path('main_vln.py').read_text())
        branch = next(n for n in source.body if isinstance(n, ast.If)
                      and '__name__' in ast.unparse(n.test))
        args = Mock(visualize=0, vln_mode='vlm_game')
        main = Mock(side_effect=CandidateRenderError('synthetic failure'))
        with patch.object(module, 'CandidateRenderer', return_value=self.renderer):
            namespace = dict(get_args=lambda: args, Queue=Mock, main=main,
                             logging=Mock(), sys=Mock(), threading=Mock(), os=Mock())
            exec(compile(ast.Module(body=branch.body, type_ignores=[]), '<headless-main>', 'exec'), namespace)
        main.assert_called_once()
        self.assertIs(main.call_args.args[-1], self.renderer)
        self.assertIsNone(self.renderer._renderer)
        namespace['sys'].exit.assert_called_once_with(1)
        namespace['threading'].Thread.assert_not_called()


if __name__ == '__main__':
    unittest.main()
