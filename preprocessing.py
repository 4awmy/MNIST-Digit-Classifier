"""Convert a bright-on-dark canvas drawing to the models' MNIST input contract."""
import base64  # To decode the Base64 data URL string emitted by the HTML5 drawing canvas
import binascii  # To catch low-level binary/ASCII decoding exceptions from malformed canvas payloads
import io  # To wrap decoded in-memory bytes into a stream that PIL can open as an image

import numpy as np  # High-performance array operations for pixel indexing, center-of-mass calculation, and float conversion
from PIL import Image  # Python Imaging Library for alpha-compositing, bounding-box cropping, resizing, and padding


class DrawingError(ValueError):
    """A drawing cannot be prepared for prediction."""


def decode_canvas(data_url):
    if not isinstance(data_url, str) or not data_url.startswith('data:image/png;base64,'):
        raise DrawingError('Draw a digit before predicting.')
    payload = data_url.split(',', 1)[1]
    if len(payload) > 3_000_000:
        raise DrawingError('The drawing is too large. Clear the canvas and try again.')
    try:
        raw = base64.b64decode(payload, validate=True)
        with Image.open(io.BytesIO(raw)) as image:
            if image.width * image.height > 1_000_000:
                raise DrawingError('The drawing dimensions are too large.')
            image.load()
            return image.copy()
    except (binascii.Error, OSError, ValueError) as exc:
        raise DrawingError('The drawing could not be read. Clear the canvas and try again.') from exc


def prepare_digit(image):
    """Return a centered float32 (28,28) image in [0,1], without dividing twice."""
    image = image.convert('RGBA')
    background = Image.new('RGBA', image.size, (0, 0, 0, 255))
    gray = Image.alpha_composite(background, image).convert('L')
    pixels = np.asarray(gray)
    foreground = pixels > 15
    if np.count_nonzero(foreground) < 10:
        raise DrawingError('The canvas is blank. Draw one digit from 0 to 9.')
    y, x = np.nonzero(foreground)
    crop = gray.crop((int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1))
    scale = 20 / max(crop.size)
    size = (max(1, round(crop.width * scale)), max(1, round(crop.height * scale)))
    crop = crop.resize(size, Image.Resampling.LANCZOS)
    square = Image.new('L', (28, 28), 0)
    square.paste(crop, ((28-size[0])//2, (28-size[1])//2))
    values = np.asarray(square, dtype=np.float32)
    yy, xx = np.indices((28, 28))
    mass = float(values.sum())
    if mass <= 0:
        raise DrawingError('The drawing is too faint. Draw a thicker digit.')
    dx = round(13.5 - float((values * xx).sum() / mass))
    dy = round(13.5 - float((values * yy).sum() / mass))
    centered = Image.new('L', (28, 28), 0)
    centered.paste(square, (dx, dy))
    return np.asarray(centered, dtype=np.float32) / np.float32(255.0)


def model_input(digit, model_name):
    digit = np.asarray(digit, dtype=np.float32)
    if digit.shape != (28, 28) or not np.isfinite(digit).all() or digit.min() < 0 or digit.max() > 1:
        raise DrawingError('Expected a normalized 28×28 digit.')
    if model_name == 'CNN':
        return digit.reshape(1, 28, 28, 1)
    if model_name in ('MLP', 'Random Forest'):
        return digit.reshape(1, 784)
    raise ValueError('Unknown model selection.')
