"""Combine measured model results without inventing or selecting results for the team."""
import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS = ('rf', 'mlp', 'cnn')
COLUMNS = (
    'model', 'selected_trial', 'validation_accuracy', 'test_accuracy', 'macro_f1',
    'selected_trial_train_seconds', 'total_search_train_seconds', 'load_seconds',
    'test_inference_seconds_10000_after_warmup', 'file_size_mib', 'parameters',
    'n_estimators', 'tree_nodes',
)


def collect_results(project_root):
    metrics = []
    metadata = []
    missing = []
    for name in MODELS:
        metrics_path = project_root / 'results' / name / 'metrics.json'
        metadata_path = project_root / 'models' / f'{name}_metadata.json'
        for path in (metrics_path, metadata_path):
            if not path.is_file():
                missing.append(str(path.relative_to(project_root)))
        if metrics_path.is_file() and metadata_path.is_file():
            metrics.append(json.loads(metrics_path.read_text(encoding='utf-8')))
            metadata.append(json.loads(metadata_path.read_text(encoding='utf-8')))
    if missing:
        raise ValueError('Complete all three local notebook runs first. Missing: ' + ', '.join(missing))
    reference = metadata[0]
    for item in metadata[1:]:
        for key in ('dataset', 'split'):
            if item[key] != reference[key]:
                raise ValueError(f'{key} differs between models; review metadata before comparing.')
        for key in ('normalization', 'dtype', 'polarity'):
            if item['preprocessing'][key] != reference['preprocessing'][key]:
                raise ValueError(f'Preprocessing {key} differs between models.')
    expected_names = {'rf': 'Random Forest', 'mlp': 'MLP', 'cnn': 'CNN'}
    for name, row, meta in zip(MODELS, metrics, metadata):
        if row.get('model') != expected_names[name]:
            raise ValueError(f'Unexpected model name in results/{name}/metrics.json.')
        if meta['metrics'] != row:
            raise ValueError(f"Metrics differ from model metadata for {row.get('model')}; files may come from different runs.")
        for key in ('validation_accuracy', 'test_accuracy', 'macro_f1'):
            if not isinstance(row[key], (int, float)) or not 0 <= row[key] <= 1:
                raise ValueError(f'Invalid {key} in {row.get("model")}.')
    warnings = []
    for key in ('python', 'tensorflow', 'scikit-learn', 'platform', 'gpu_devices'):
        values = [json.dumps(item['environment'].get(key), sort_keys=True) for item in metadata]
        if len(set(values)) > 1:
            warnings.append(f'{key} differs: ' + '; '.join(f'{name}={value}' for name, value in zip(MODELS, values)))
    warnings.append('Random Forest uses CPU even if GPU devices are available. Qualify timing comparisons.')
    warnings.append('10,000-image timing is batched inference, not single-drawing app response time.')
    warnings.append('RF tree/node counts and neural-network parameter counts are different complexity measures.')
    return metrics, warnings


def write_comparison(project_root):
    rows, warnings = collect_results(project_root)
    folder = project_root / 'results'
    with (folder / 'comparison.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# Measured model comparison', '',
             '| Model | Validation accuracy | Test accuracy | Macro F1 | Selected fit (s) | Size (MiB) |',
             '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['model']} | {row['validation_accuracy']:.2%} | {row['test_accuracy']:.2%} | "
                     f"{row['macro_f1']:.4f} | {row['selected_trial_train_seconds']:.2f} | {row['file_size_mib']:.2f} |")
    lines += ['', 'Context for interpreting the table:', ''] + ['- ' + message for message in warnings]
    (folder / 'comparison.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (folder / 'comparison_warnings.txt').write_text('\n'.join(warnings) + '\n', encoding='utf-8')
    return folder / 'comparison.csv'


if __name__ == '__main__':
    try:
        path = write_comparison(PROJECT_ROOT)
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise SystemExit(f'Comparison not generated: {exc}')
    print('Created:', path)
