"""Exercise all real saved artifacts through the application's CPU input contract."""
import json
import os
from pathlib import Path
import sys
import time
import platform
from importlib.metadata import version, PackageNotFoundError

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '4')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '2')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
from tensorflow.keras.datasets import mnist
from model_io import MODEL_FILES, load_artifact, predict_digit
from preprocessing import prepare_digit


def main():
    (_, _), (images, labels) = mnist.load_data()
    indices = [int(np.flatnonzero(labels == digit)[0]) for digit in range(10)]
    drawings = [prepare_digit(Image.fromarray(images[i]).resize((280, 280))) for i in indices]
    evidence = {'device': 'CPU', 'sample': 'First test example of each digit, enlarged and passed through drawing preprocessing',
                'note': 'Ten examples are an integration smoke check, not a drawing accuracy benchmark.',
                'tested_unix': time.time(), 'environment': {'python': platform.python_version()}, 'models': []}
    for package in ['tensorflow', 'tensorflow-cpu', 'keras', 'scikit-learn', 'numpy']:
        try:
            evidence['environment'][package] = version(package)
        except PackageNotFoundError:
            evidence['environment'][package] = None
    for name, filename in MODEL_FILES.items():
        start = time.perf_counter()
        model = load_artifact(name, ROOT / 'models' / filename)
        load_seconds = time.perf_counter() - start
        predict_digit(model, name, drawings[0])  # Warm up before timing.
        samples = []
        for digit, drawing in enumerate(drawings):
            result = predict_digit(model, name, drawing)
            assert 0 <= result['digit'] <= 9
            assert len(result['scores']) == 10 and np.isclose(sum(result['scores']), 1)
            samples.append({'true_digit': digit, 'predicted_digit': result['digit'], 'milliseconds': result['milliseconds']})
        timings = [predict_digit(model, name, drawing)['milliseconds'] for _ in range(10) for drawing in drawings]
        evidence['models'].append({'model': name, 'load_seconds': load_seconds,
            'median_milliseconds': float(np.median(timings)), 'p95_milliseconds': float(np.percentile(timings, 95)),
            'correct_of_10_smoke_examples': sum(s['true_digit'] == s['predicted_digit'] for s in samples), 'samples': samples})
        print(f'{name}: loaded, ten valid class-score vectors; warmed median {np.median(timings):.2f} ms', flush=True)
    target = ROOT / 'results' / 'app_smoke_check.json'
    target.write_text(json.dumps(evidence, indent=2) + '\n', encoding='utf-8')
    print('Saved', target)


if __name__ == '__main__':
    main()
