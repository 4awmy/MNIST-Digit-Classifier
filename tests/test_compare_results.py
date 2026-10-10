import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('comparison', PROJECT / 'tools' / 'compare_results.py')
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for name, title in [('rf', 'Random Forest'), ('mlp', 'MLP'), ('cnn', 'CNN')]:
            folder = self.root / 'results' / name
            folder.mkdir(parents=True)
            (self.root / 'models').mkdir(exist_ok=True)
            metrics = {
                'model': title, 'selected_trial': 'A', 'validation_accuracy': 0.90,
                'test_accuracy': 0.91, 'macro_f1': 0.90,
                'selected_trial_train_seconds': 1.0, 'total_search_train_seconds': 3.0,
                'load_seconds': 0.1, 'test_inference_seconds_10000_after_warmup': 0.5,
                'file_size_mib': 1.0, 'parameters': None if name == 'rf' else 100,
            }
            metadata = {
                'dataset': 'MNIST',
                'split': {'train': 50000, 'validation': 10000, 'test': 10000, 'random_state': 42, 'stratify': True},
                'preprocessing': {'normalization': 'raw pixel / 255.0', 'dtype': 'float32',
                                  'polarity': 'bright digit on dark background'},
                'environment': {'python': 'example', 'tensorflow': 'example',
                                'scikit-learn': 'example', 'platform': 'example', 'gpu_devices': []},
                'metrics': metrics,
            }
            (folder / 'metrics.json').write_text(json.dumps(metrics), encoding='utf-8')
            (self.root / 'models' / f'{name}_metadata.json').write_text(json.dumps(metadata), encoding='utf-8')

    def tearDown(self):
        self.temp.cleanup()

    def update_metadata(self, name, mutate):
        path = self.root / 'models' / f'{name}_metadata.json'
        data = json.loads(path.read_text(encoding='utf-8'))
        mutate(data)
        path.write_text(json.dumps(data), encoding='utf-8')

    def test_complete_results_create_table(self):
        path = comparison.write_comparison(self.root)
        self.assertTrue(path.is_file())
        self.assertIn('Random Forest', (path.parent / 'comparison.md').read_text(encoding='utf-8'))
        self.assertEqual(len(path.read_text(encoding='utf-8').splitlines()), 4)

    def test_missing_run_is_rejected_without_partial_table(self):
        (self.root / 'results' / 'cnn' / 'metrics.json').unlink()
        with self.assertRaisesRegex(ValueError, 'Missing:'):
            comparison.write_comparison(self.root)
        self.assertFalse((self.root / 'results' / 'comparison.csv').exists())

    def test_different_split_is_rejected(self):
        self.update_metadata('mlp', lambda data: data['split'].update(train=60000))
        with self.assertRaisesRegex(ValueError, 'split differs'):
            comparison.collect_results(self.root)

    def test_different_normalization_is_rejected(self):
        self.update_metadata('cnn', lambda data: data['preprocessing'].update(normalization='none'))
        with self.assertRaisesRegex(ValueError, 'normalization differs'):
            comparison.collect_results(self.root)

    def test_stale_metrics_are_rejected(self):
        self.update_metadata('mlp', lambda data: data['metrics'].update(test_accuracy=0.50))
        with self.assertRaisesRegex(ValueError, 'different runs'):
            comparison.collect_results(self.root)

    def test_environment_difference_is_reported(self):
        self.update_metadata('cnn', lambda data: data['environment'].update(gpu_devices=['example_gpu']))
        rows, warnings = comparison.collect_results(self.root)
        self.assertEqual(len(rows), 3)
        self.assertTrue(any('gpu_devices differs' in warning for warning in warnings))


if __name__ == '__main__':
    unittest.main()
