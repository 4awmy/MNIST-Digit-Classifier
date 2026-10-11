"""Build the final narrative only from successfully completed local experiment evidence."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
names = {'rf': 'Random Forest', 'mlp': 'MLP', 'cnn': 'CNN'}
metrics = {key: json.loads((ROOT / 'results' / key / 'metrics.json').read_text()) for key in names}
metadata = {key: json.loads((ROOT / 'models' / f'{key}_metadata.json').read_text()) for key in names}
trials = {key: list(csv.DictReader((ROOT / 'results' / key / 'tuning_results.csv').open())) for key in names}
app = json.loads((ROOT / 'results' / 'app_smoke_check.json').read_text())
browser = json.loads((ROOT / 'results' / 'app_browser' / 'checks.json').read_text())
assert browser['status'] == 'passed'
assert app['environment']['tensorflow-cpu'] == metadata['cnn']['environment']['tensorflow'], 'Verify the clean CPU app environment first.'
for key in names:
    assert json.loads((ROOT / 'results' / key / 'run_status.json').read_text())['status'] == 'complete'
    assert metrics[key] == metadata[key]['metrics']
    finished = json.loads((ROOT / 'results' / key / 'run_status.json').read_text())['finished_unix']
    assert app['tested_unix'] >= finished and browser['tested_unix'] >= finished, 'Refresh app checks after the latest training run.'

def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] + ['| ' + ' | '.join(map(str,row)) + ' |' for row in rows])

def pct(value): return f'{float(value):.2%}'
def num(value): return f'{float(value):.2f}'

report = ['# Assignment 1: MNIST digit recognition', '',
    'All three model families were trained locally from fresh kernels. This report uses the corrected runs and saved artifacts, with the original team study notes retained in the notebooks. The notebooks remain separate and feed one shared Streamlit application.', '',
    '## Data and preprocessing', '',
    'MNIST provides 60,000 development images and a separate 10,000-image test set. All models use the same stratified 50,000/10,000 training/validation split, with seed 42. Pixels are cast to float32 and divided by 255 once. RF/MLP receive 784 features; CNN receives 28 by 28 by 1 images. Labels remain integer digits; sparse categorical cross-entropy is used by the networks. The test set is scored after configuration selection.', '',
    '## Hyperparameter experiments', '',
    '### Random Forest', '',
    table(['Trial','Trees','Max depth','Train accuracy','Validation accuracy','Fit seconds'],
      [[r['name'],r['n_estimators'],r['max_depth'] or 'Unlimited',pct(r['train_accuracy']),pct(r['val_accuracy']),num(r['train_seconds'])] for r in trials['rf']]), '',
    'A to B changes tree count; B to C changes the depth limit. RF uses seed 42 and all CPU cores. Select highest validation accuracy, then fewer trees, shallower depth and trial name. Selected configuration: `' + json.dumps(metadata['rf']['hyperparameters']) + '`.', '',
    '### Fully connected MLP', '',
    table(['Trial','Hidden units','Learning rate','Batch','L2','Best epoch','Validation accuracy','Fit seconds'],
      [[r['name'],r['hidden_units'],r['learning_rate'],r['batch_size'],r['l2'],r['best_epoch'],pct(r['val_accuracy']),num(r['train_seconds'])] for r in trials['mlp']]), '',
    'Each alternative changes one baseline setting: width, learning rate, batch size, or L2. The network uses Dense/ReLU hidden layers and ten softmax outputs. Training allows 20 epochs with validation-loss early stopping, patience 3. Each trial contributes its minimum-validation-loss checkpoint; select by validation accuracy, then loss, parameter count and trial name. L2 contributes to reported loss, so losses across differing penalties are not pure predictive cross-entropy comparisons. Selected configuration: `' + json.dumps(metadata['mlp']['hyperparameters']) + '`.', '',
    '### Introductory CNN', '',
    table(['Trial','Filters','Learning rate','Batch','Dense units','Best epoch','Validation accuracy','Fit seconds'],
      [[r['name'],r['filters'],r['learning_rate'],r['batch_size'],r['dense_units'],r['best_epoch'],pct(r['val_accuracy']),num(r['train_seconds'])] for r in trials['cnn']]), '',
    'The CNN uses two 3 by 3 convolution/ReLU stages, each with 2 by 2 max pooling, then flatten, Dense(64, ReLU), and ten softmax outputs. Trials change filter count or learning rate. The same checkpoint, early-stopping and validation-ranking policy applies. No pretrained backbone is used. Selected configuration: `' + json.dumps(metadata['cnn']['hyperparameters']) + '`.', '',
    '## What the experiments show', '',
    'Validation accuracy changes below are relative to each family\'s baseline A. They explain the current selection, not a statistically established advantage. Keeping one changed setting per alternative makes the comparison easier to interpret.', '',
    table(['Family','Trial','Validation change vs A (points)','Fit time vs A'],
      [[names[k],r['name'],f"{(float(r['val_accuracy'])-float(trials[k][0]['val_accuracy']))*100:+.2f}",f"{float(r['train_seconds'])/float(trials[k][0]['train_seconds']):.2f}x"]
       for k in names for r in trials[k][1:]]), '',
    'A wider MLP increases parameter count; lower learning rate changes the optimization pace; larger batches reduce updates per epoch; L2 penalizes large weights. More RF trees increase training/storage cost, and removing the depth cap permits larger trees. Wider CNN filters increase capacity while retaining the same introductory two-stage structure. The selected values are justified by the validation ranking above, with test performance, artifact size and timings providing the final comparison.', '',
    '## Final saved-model results', '',
    table(['Model','Validation accuracy','Test accuracy','Macro precision','Macro recall','Macro F1','Size MiB'],
      [[names[k],pct(m['validation_accuracy']),pct(m['test_accuracy']),f"{m['macro_precision']:.4f}",f"{m['macro_recall']:.4f}",f"{m['macro_f1']:.4f}",num(m['file_size_mib'])] for k,m in metrics.items()]), '',
    table(['Model','Selected fit s','Total search fit s','Reload s','10,000-image inference s','Complexity'],
      [[names[k],num(m['selected_trial_train_seconds']),num(m['total_search_train_seconds']),num(m['load_seconds']),num(m['test_inference_seconds_10000_after_warmup']),
        f"{m['parameters']:,} parameters" if m.get('parameters') is not None else f"{m['n_estimators']} trees; {m['tree_nodes']:,} nodes"] for k,m in metrics.items()]), '',
    'Hardware: Intel Core i7-13650HX, 20 logical CPUs, 32 GB installed RAM, and NVIDIA RTX 4060 Laptop GPU (8 GB). Ubuntu 24.04/WSL2 used Python ' + metadata['cnn']['environment']['python'] + ', TensorFlow ' + metadata['cnn']['environment']['tensorflow'] + ', Keras ' + metadata['cnn']['environment']['keras'] + ', and scikit-learn ' + metadata['rf']['environment']['scikit-learn'] + '. RF trained/inferred on CPU; MLP/CNN training and notebook batch inference used GPU. Timing includes checkpoint activity and is specific to this laptop. Tree/node counts and network parameter counts are different units.', '',
    '## Generalization and errors', '']

for key in names:
    chosen = next(r for r in trials[key] if r['name'] == metrics[key]['selected_trial'])
    train_key = 'train_accuracy' if key == 'rf' else 'train_accuracy_at_best_epoch'
    gap = float(chosen[train_key]) - float(chosen['val_accuracy'])
    cm = list(csv.reader((ROOT / 'results' / key / 'confusion_matrix.csv').open()))
    confusions = sorted([(int(cm[i+1][j+1]), i, j) for i in range(10) for j in range(10) if i != j], reverse=True)[:3]
    errors = round((1-metrics[key]['test_accuracy'])*10000)
    report += [f"**{names[key]}:** {errors} errors out of 10,000. {'Training accuracy' if key == 'rf' else 'Selected-epoch training accuracy'} {pct(chosen[train_key])}, validation accuracy {pct(chosen['val_accuracy'])}, gap {gap*100:.2f} percentage points. The largest directed confusions were " + ', '.join(f'{i} predicted as {j} ({n})' for n,i,j in confusions) + '.', '']
    report += [f"Evidence: [confusion matrix](../results/{key}/confusion_matrix.png), [per-digit metrics](../results/{key}/classification_report.csv), [example errors](../results/{key}/misclassified_examples.png).", '']
    if key in ('mlp','cnn'):
        report += [f"[Training and validation learning curves](../results/{key}/learning_curves.png).", '']
report += ['Training-epoch neural accuracy averages predictions made while weights change; validation accuracy uses the checkpoint weights, so this gap is descriptive rather than an exact matched-state measurement. Early stopping limits continued fitting after validation loss deteriorates. These small controlled searches use one seed per setting; small differences require repeated-seed experiments before claiming a reliable improvement.', '',
    '## Model recommendation and app responsiveness', '']
winner = max(metrics, key=lambda k: metrics[k]['validation_accuracy'])
winner_app = next(m for m in app['models'] if m['model'] == names[winner])
report += [f"Recommend **{names[winner]}** using validation accuracy ({pct(metrics[winner]['validation_accuracy'])}); its separate test result is {pct(metrics[winner]['test_accuracy'])} with macro F1 {metrics[winner]['macro_f1']:.4f}. This selection uses validation evidence; the final test table provides independent performance evidence. " +
    ('The CNN uses spatial filters and shared weights suited to handwritten images. ' if winner=='cnn' else '') +
    f"Its saved file is {metrics[winner]['file_size_mib']:.2f} MiB. All three models remain selectable, making accuracy, complexity and response cost visible.", '',
    'After selection, the test comparison shows ' + '; '.join(f"{(metrics[winner]['test_accuracy']-metrics[k]['test_accuracy'])*100:+.2f} percentage points versus {names[k]}" for k in names if k != winner) + '. The cost is assessed alongside that gain: ' + '; '.join(f"{names[k]} selected fit {metrics[k]['selected_trial_train_seconds']:.2f}s, artifact {metrics[k]['file_size_mib']:.2f} MiB" for k in names) + '. Training times use different CPU/GPU devices, so they describe this workflow rather than equal-hardware speed. The generalization gaps and error patterns above qualify the accuracy comparison.', '',
    f"On the local CPU app check, the recommended model's warmed single-drawing median was {winner_app['median_milliseconds']:.2f}ms. This supports the intended one-digit interaction locally; Cloud responsiveness must still be checked after deployment. Parameter counts, file sizes and measured response times describe different costs and should be considered together.", '',
    table(['Model','CPU load s','Warmed single-drawing median ms','p95 ms'], [[m['model'],num(m['load_seconds']),num(m['median_milliseconds']),num(m['p95_milliseconds'])] for m in app['models']]), '',
    'The app timings use 100 warmed calls per model on CPU. They measure model execution after preprocessing and exclude browser/network round trips. Ten enlarged MNIST examples were also passed through the complete drawing-preparation pipeline as a smoke check; this is not a handwriting accuracy benchmark. Cloud startup/memory/latency require measurement on the actual deployment.', '',
    '## Interactive application and verification', '',
    'Run `python -m streamlit run app.py`. Draw one white digit on the black canvas with a mouse or finger, select CNN/MLP/Random Forest, then press Predict. Clear resets drawing and result. Changing the model keeps the drawing and removes the previous prediction. The preview shows the grayscale image after foreground cropping, aspect-preserving resizing to a 20-pixel bounding box, centering in 28 by 28, center-of-mass adjustment, and float32 normalization. RF/MLP flatten this array; CNN adds a channel. Blank input is rejected. Model scores are not calibrated confidence.', '',
    'Browser checks passed for blank input, mouse/touch drawing, all three model choices, drawing persistence, stale-result invalidation, Clear, processed preview and comparison display. Evidence and screenshots are in `results/app_browser/`; saved-model checks are in `results/app_smoke_check.json`. Automated preprocessing/comparison checks also pass. Saved-artifact reload consistency was verified by each training notebook.', '',
    '![Local drawing application](../results/app_browser/desktop.png)', '',
    '## Reproducibility, collaboration and limitations', '',
    'The public repository is [4awmy/MNIST-Digit-Classifier](https://github.com/4awmy/MNIST-Digit-Classifier). The team has sent its collaborator invitation. The notebooks, source code, final artifacts, metrics, environment records and dependency locks support review and reproduction; backups and intermediate checkpoints are excluded from Git. Team names are omitted as requested.', '',
    'See [methodology](methodology.md), [local setup](local_setup.md), [training changes](../README.md#training-changes), and [deployment guide](deployment.md). The user will deploy after the GitHub update. Record the final Cloud URL only after a successful live deployment and prediction check. Local success does not establish Cloud success. Pushing and deployment load the saved artifacts; they do not execute training.', '',
    'MNIST is a clean centered single-digit dataset. Arbitrary drawing styles and photographs can differ substantially; this app does not segment multiple digits. Repeated seeds, broader validation-driven searches and a labelled drawing evaluation set would strengthen the evidence. Picture upload remains the next optional feature after deployment and final documentation review.']
(ROOT / 'docs' / 'report.md').write_text('\n'.join(report)+'\n', encoding='utf-8')

summary = table(['Model','Selected trial','Validation accuracy','Test accuracy','Macro F1'], [[names[k],m['selected_trial'],pct(m['validation_accuracy']),pct(m['test_accuracy']),f"{m['macro_f1']:.4f}"] for k,m in metrics.items()])
readme = f'''# MNIST Digit Classifier

Train and compare Random Forest, MLP and CNN models, then draw a digit in a Streamlit app and choose a saved model for prediction.

## Run the app

Use Python 3.12. From the repository root:

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open **http://localhost:8501**. Draw one digit from 0 to 9 using your mouse or finger, select a model, and click **Predict**. **Clear** resets the canvas. **What the model sees** previews the prepared 28 by 28 input.

![Drawing application](results/app_browser/desktop.png)

## Results

{summary}

Recommended model: **{names[winner]}**, selected using validation accuracy. The [report](docs/report.md) explains the experiments, final settings, errors, complexity and timing. The [methodology](docs/methodology.md) describes data preparation and model selection.

## Project layout

Keep the three notebooks separate. Each trains one model family; the app loads all three saved models.

| Training notebook | Trials | Final artifact |
|---|---:|---|
| Random_forest_.ipynb | 3 | models/rf_final.joblib |
| Mlp.ipynb | 5 | models/mlp_final.keras |
| CNN.ipynb | 3 | models/cnn_final.keras |

`app.py` starts Streamlit. `assets/` and `canvas_component.py` provide the drawing canvas; `preprocessing.py` prepares the image; `model_io.py` loads the models. `results/` holds recorded training outputs, metrics and plots. The original assignment brief is [Assignment 1.md](Assignment%201.md).

The root notebooks have been rearranged into smaller cells closer to the original layout, keeping the original study notes with brief factual corrections. Their outputs are cleared so they do not imply the rearranged cells were rerun. Previously executed notebooks in `results/<model>/executed.ipynb` are unchanged snapshots of the completed training runs, including the notes that existed at that time.

## Training changes

The original RF and CNN notebooks already had useful validation splits and experiments. The original MLP's one-hot labels with categorical cross-entropy were valid too.

- **MLP:** added the shared stratified 50,000/10,000 training/validation split and five trials covering width, learning rate, batch size and L2. Uses Dense computational layers, integer labels with sparse cross-entropy, validation checkpoints and early stopping. Removing the original Dropout follows the assignment's literal fully-connected-only wording; Dropout itself was not an error. Changing the label representation alone does not improve accuracy.
- **RF:** kept the three tree/depth experiments and compressed saving; added automatic validation-based selection and consistent evaluation.
- **CNN:** kept the simple two-stage architecture and three configurations; fixed best-only checkpoint saving and added early stopping and automatic selection.
- **All three:** use float32 pixels divided by 255 once, consistent evaluation, saved final models and reload checks. RF/MLP use 784 features; CNN uses 28 by 28 by 1.

We retrained to produce saved models and measured results matching those training changes. **This notebook/documentation cleanup does not retrain or change the saved models. Pushing to GitHub and deploying do not retrain either:** the app loads the existing artifacts. Changing training data, architecture or training settings requires rerunning the affected experiments.

For deliberate retraining, environment setup and laptop-specific launch commands, see [local setup](docs/local_setup.md). App dependencies are in `requirements.txt`; training, GPU and optional browser-check dependencies are separate. Dependency locks record the working environments.

## Verification

```powershell
python -m unittest discover -s tests -v
python tools/check_saved_models.py
python tools/compare_results.py
```

The saved-model check uses ten enlarged MNIST samples through drawing preprocessing; this is an integration check, not a handwriting accuracy benchmark. Optional mouse/touch browser checks are described in [local setup](docs/local_setup.md). After retraining, refresh saved-model and browser checks before running `python tools/build_report.py` to regenerate the report and this README.

## GitHub and next steps

Repository: [4awmy/MNIST-Digit-Classifier](https://github.com/4awmy/MNIST-Digit-Classifier). The [archive/full-project branch](https://github.com/4awmy/MNIST-Digit-Classifier/tree/archive/full-project) preserves the full version before cleanup, including extra explanations and documents. Backups, environments and intermediate checkpoints stay outside Git.

1. Review the notebooks, methodology and report so the team can explain the experiments and model choice. Confirm the collaborator invitation was accepted.
2. Deploy `main` / `app.py` with Python 3.12 using the [Streamlit deployment guide](docs/deployment.md). The app uses CPU inference and the committed model files.
3. Test drawing, Predict, Clear, all three models, touch input and the processed preview on the live app. Then add its public URL here and to the report.
4. Add picture upload afterward, reusing the image preparation and saved models. Start with one clear digit per image and validate representative uploads.
'''
(ROOT / 'README.md').write_text(readme, encoding='utf-8')
print('Completed report and README from real local evidence.')
