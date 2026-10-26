"""Synchronous candidate-only rendering; no window or GUI update thread."""
import threading

import numpy as np
import open3d as o3d
from open3d.visualization import rendering

from utils.candidate_images import (
    CandidateRenderError, compose_candidate_image, publish_candidate, validate_first_view)


class CandidateRenderer:
    # Captured from the untouched 1280x800 GUI on this machine (font size 32).
    WIDTH = 640
    HEIGHT = 800
    FOV = 90.0
    NEAR = 0.1

    def __init__(self):
        self._owner = threading.get_ident()
        self._renderer = rendering.OffscreenRenderer(self.WIDTH, self.HEIGHT)
        self._renderer.scene.set_background([1, 1, 1, 1])
        self._renderer.scene.show_axes(True)

    @classmethod
    def configuration(cls):
        return dict(backend='open3d_offscreen_sync_v1', reference='gui_640x800_fov90_v1',
                    width=cls.WIDTH, height=cls.HEIGHT, vertical_fov=cls.FOV,
                    near_clip=cls.NEAR, camera_offset=[-0.5, 4, 0], up=[0, 1, 0],
                    clipping_bounds=[[-5, -5, -5], [5, 5, 5]],
                    open3d_version=o3d.__version__,
                    timeout_kind='synchronous native call; no enforceable render deadline')

    def _check_open(self):
        if threading.get_ident() != self._owner:
            raise CandidateRenderError('Offscreen renderer must be used on its creating thread')
        if self._renderer is None:
            raise CandidateRenderError('Offscreen renderer is closed')

    def render(self, request, points, colors, bbox_np, first_view):
        self._check_open()
        validate_first_view(request, first_view)
        try:
            scene = self._renderer.scene
            scene.clear_geometry()
            cloud = o3d.t.geometry.PointCloud(o3d.core.Tensor(np.asarray(points, dtype=np.float32)))
            cloud.point.colors = o3d.core.Tensor(np.asarray(colors, dtype=np.float32))
            material = rendering.MaterialRecord()
            material.shader = 'defaultUnlit'
            scene.add_geometry('full_pcd', cloud, material)
            bounds = np.asarray(bbox_np, dtype=np.float32)
            bbox = o3d.geometry.AxisAlignedBoundingBox(bounds[0], bounds[1])
            bbox.color = [1, 0, 0]
            line = rendering.MaterialRecord()
            line.shader = 'unlitLine'
            line.line_width = 5.0
            scene.add_geometry(f'bbox_{request.candidate_id}', bbox, line)
            center = bbox.get_center()
            eye = center + [-0.5, 4, 0]
            # Match SceneWidget.look_at() using init_render's fixed [-5,5]^3
            # bounds, rather than OffscreenRenderer's current-geometry default.
            eye32 = eye.astype(np.float32).astype(np.float64)
            far = max(1.0, 2.5 * np.linalg.norm(eye32) + 4.0 * np.linalg.norm([10, 10, 10]))
            self._renderer.setup_camera(self.FOV, center, eye, [0, 1, 0], self.NEAR, far)
            image = self._renderer.render_to_image()
            return publish_candidate(request, compose_candidate_image(image, request.candidate_id, first_view))
        except CandidateRenderError:
            raise
        except Exception as exc:
            raise CandidateRenderError(f'Offscreen candidate {request.request_id}: {exc}') from exc

    def close(self):
        if self._renderer is not None:
            self._check_open()
            # Renderer owns its scene and destroys it before its engine.
            self._renderer = None
