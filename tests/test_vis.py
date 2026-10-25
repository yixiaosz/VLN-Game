"""Pixel-parity and warning checks for the Supervision 0.21 annotation migration."""
import dataclasses
import unittest
import warnings

import numpy as np
import supervision as sv

from utils.vis import vis_result_fast


def legacy_annotation(image, detections, classes, color, instance_random_color, draw_bbox):
    """Pre-migration drawing path, retained only as a visual test reference."""
    box = sv.BoxAnnotator(color=color, text_scale=0.3, text_thickness=1, text_padding=2)
    mask = sv.MaskAnnotator(color=color)
    labels = [f'{classes[class_id]} {confidence:0.2f}'
              for _, _, confidence, class_id, _, _ in detections]
    if instance_random_color:
        detections = dataclasses.replace(detections)
        detections.class_id = np.arange(len(detections))
    result = mask.annotate(scene=image.copy(), detections=detections)
    if draw_bbox:
        result = box.annotate(scene=result, detections=detections, labels=labels)
    return result


class AnnotationTests(unittest.TestCase):
    def setUp(self):
        self.image = np.full((240, 320, 3), (60, 110, 150), dtype=np.uint8)
        self.classes = ['cabinet', 'chair', 'table']

    def detections(self, boxes, with_masks=True):
        xyxy = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        count = len(xyxy)
        masks = np.zeros((count, 240, 320), dtype=bool) if with_masks else None
        if masks is not None:
            for mask, (x1, y1, x2, y2) in zip(masks, xyxy.astype(int)):
                mask[max(y1, 0):min(y2, 240), max(x1, 0):min(x2, 320)] = True
        return sv.Detections(xyxy=xyxy, mask=masks,
                             confidence=np.linspace(0.71, 0.99, count, dtype=np.float32),
                             class_id=np.asarray([1, 0, 1][:count], dtype=int))

    def compare(self, detections, color=sv.ColorPalette.DEFAULT,
                instance_random_color=False, draw_bbox=True):
        if not hasattr(sv, 'BoxAnnotator'):
            self.skipTest('Legacy visual reference is unavailable in this Supervision version')
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            expected = legacy_annotation(self.image, detections, self.classes, color,
                                         instance_random_color, draw_bbox)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            actual = vis_result_fast(self.image, detections, self.classes, color,
                                     instance_random_color, draw_bbox)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(len(caught), 0, [str(w.message) for w in caught])

    def test_annotation_matches_legacy_visuals(self):
        scenarios = [[], [[40, 60, 180, 180]],
                     [[40, 60, 180, 180], [60, 65, 200, 190], [30, 50, 160, 175]],
                     [[0, 1, 100, 80], [280, 210, 319, 239]]]
        for boxes in scenarios:
            for random_color in (False, True):
                for draw_bbox in (False, True):
                    with self.subTest(boxes=boxes, random_color=random_color, draw_bbox=draw_bbox):
                        self.compare(self.detections(boxes), instance_random_color=random_color,
                                     draw_bbox=draw_bbox)
        self.compare(self.detections([[40, 60, 180, 180]], with_masks=False))

    def test_scalar_color_and_custom_palette_match_legacy(self):
        detections = self.detections([[40, 60, 180, 180], [45, 55, 190, 185]])
        for color in (sv.Color(r=20, g=80, b=180),
                      sv.ColorPalette.from_hex(['#112233', '#FF7700', '#88CC55'])):
            for random_color in (False, True):
                with self.subTest(color=color, random_color=random_color):
                    self.compare(detections, color=color, instance_random_color=random_color)

    def test_input_image_and_detections_are_not_modified(self):
        detections = self.detections([[40, 60, 180, 180], [60, 65, 200, 190]])
        before_image = self.image.copy()
        before = {key: getattr(detections, key).copy() for key in ('xyxy', 'mask', 'confidence', 'class_id')}
        vis_result_fast(self.image, detections, self.classes, instance_random_color=True)
        np.testing.assert_array_equal(self.image, before_image)
        for key, values in before.items():
            np.testing.assert_array_equal(getattr(detections, key), values)

    def test_repeated_frames_emit_no_deprecated_annotator_warnings(self):
        detections = self.detections([[40, 60, 180, 180]])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            for _ in range(10):
                vis_result_fast(self.image, detections, self.classes)
        self.assertEqual(len(caught), 0, [str(w.message) for w in caught])


if __name__ == '__main__':
    unittest.main()
