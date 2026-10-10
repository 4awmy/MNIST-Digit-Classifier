"""Load project-owned artifacts and adapt predictions for the drawing app."""
from pathlib import Path
import time

import numpy as np

from preprocessing import model_input

MODEL_FILES = {'Random Forest': 'rf_final.joblib', 'MLP': 'mlp_final.keras', 'CNN': 'cnn_final.keras'}


def load_artifact(name, path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'{path.name} is missing. Finish local training first.')
    if name == 'Random Forest':
        import joblib
        model = joblib.load(path)
        if model.n_features_in_ != 784 or not np.array_equal(model.classes_, np.arange(10)):
            raise ValueError('The saved forest does not match the MNIST input contract.')
        model.n_jobs = 1  # Avoid thread startup overhead for one drawing.
    else:
        from tensorflow import keras
        model = keras.models.load_model(path, compile=False)
        expected = (None, 28, 28, 1) if name == 'CNN' else (None, 784)
        if model.input_shape != expected or model.output_shape != (None, 10):
            raise ValueError('The saved network does not match the MNIST input contract.')
    return model


def predict_digit(model, name, digit):
    inputs = model_input(digit, name)
    start = time.perf_counter()
    if name == 'Random Forest':
        raw_scores = np.asarray(model.predict_proba(inputs)[0], dtype=np.float64)
        scores = np.zeros(10, dtype=np.float64)
        scores[np.asarray(model.classes_, dtype=int)] = raw_scores
    else:
        scores = np.asarray(model(inputs, training=False))[0].astype(np.float64)
    milliseconds = (time.perf_counter() - start) * 1000
    if scores.shape != (10,) or not np.isfinite(scores).all() or (scores < 0).any() or scores.sum() <= 0:
        raise ValueError('The model returned invalid prediction scores.')
    scores = scores / scores.sum()
    return {'digit': int(scores.argmax()), 'scores': scores.tolist(), 'milliseconds': milliseconds, 'model': name}
