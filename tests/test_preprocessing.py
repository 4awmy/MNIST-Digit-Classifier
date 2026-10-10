import base64
import io
from pathlib import Path
import sys
import unittest

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocessing import DrawingError, decode_canvas, model_input, prepare_digit


class DrawingTests(unittest.TestCase):
    def stroke(self, mode='RGB'):
        image = Image.new(mode, (280, 280), (0, 0, 0, 0) if mode == 'RGBA' else (0, 0, 0))
        ImageDraw.Draw(image).line([(40, 30), (70, 50), (70, 220)], fill='white', width=18)
        return image

    def test_blank_is_rejected(self):
        with self.assertRaises(DrawingError):
            prepare_digit(Image.new('RGB', (280, 280), 'black'))

    def test_transparent_blank_is_rejected(self):
        with self.assertRaises(DrawingError):
            prepare_digit(Image.new('RGBA', (280, 280), (255, 255, 255, 0)))

    def test_normalization_and_centering(self):
        digit = prepare_digit(self.stroke())
        self.assertEqual(digit.shape, (28, 28))
        self.assertEqual(digit.dtype, np.float32)
        self.assertGreater(digit.max(), .9)
        self.assertLessEqual(digit.max(), 1)
        yy, xx = np.indices(digit.shape)
        self.assertLess(abs((digit * xx).sum()/digit.sum() - 13.5), .6)
        self.assertLess(abs((digit * yy).sum()/digit.sum() - 13.5), .6)

    def test_transparency_composites_on_black(self):
        np.testing.assert_allclose(prepare_digit(self.stroke('RGBA')), prepare_digit(self.stroke()))

    def test_aspect_ratio_preserved_for_narrow_digit(self):
        image = Image.new('L', (280, 280), 0)
        ImageDraw.Draw(image).rectangle((80, 20, 96, 220), fill=255)
        y, x = np.nonzero(prepare_digit(image) > .1)
        self.assertLess(x.max()-x.min(), y.max()-y.min())

    def test_model_shapes_without_second_normalization(self):
        digit = prepare_digit(self.stroke())
        for name, shape in [('MLP', (1, 784)), ('Random Forest', (1, 784)), ('CNN', (1, 28, 28, 1))]:
            inputs = model_input(digit, name)
            self.assertEqual(inputs.shape, shape)
            self.assertEqual(inputs.max(), digit.max())

    def test_png_round_trip_and_invalid_input(self):
        buffer = io.BytesIO()
        self.stroke().save(buffer, format='PNG')
        decoded = decode_canvas('data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode())
        self.assertEqual(decoded.size, (280, 280))
        for value in (None, 'not an image', 'data:image/png;base64,!!!'):
            with self.assertRaises(DrawingError):
                decode_canvas(value)


if __name__ == '__main__':
    unittest.main()
