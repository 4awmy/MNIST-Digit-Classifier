# MNIST Digit Classifier

Train and compare a Random Forest, a fully connected MLP, and an introductory CNN, then draw a digit in a Streamlit app and choose any saved model for prediction.

All three models were trained locally under WSL2. RF uses CPU; MLP/CNN used the RTX 4060. The app, saved-model integration checks, and mouse/touch browser checks pass. Cloud deployment status is tracked in [deployment](docs/deployment.md).

## Measured results

| Model | Selected trial | Validation accuracy | Test accuracy | Macro F1 |
| --- | --- | --- | --- | --- |
| Random Forest | C | 96.65% | 96.95% | 0.9693 |
| MLP | B | 97.47% | 97.62% | 0.9761 |
| CNN | B | 98.74% | 99.02% | 0.9901 |

Recommended model: **CNN**, selected using validation evidence. See the [report](docs/report.md) for configuration choices, errors, complexity, timing and limitations. Scores are from the corrected local runs, not the historical notebook outputs.

## Run the app

Use Python 3.12. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

On the configured laptop, `.\tools\run_app.ps1` uses the existing WSL environment. Open **http://localhost:8501**. Draw one digit from 0 to 9 using your mouse or finger, select a model, and click Predict. Clear resets the canvas. “What the model sees” shows the prepared 28 by 28 input. Picture upload is deferred.

![Drawing application](results/app_browser/desktop.png)

## Project layout

The three notebooks remain independent. Each saves one model; one app loads all three. Desktop originals are at the root; a packaged copy may use `notebooks/`. Both notebook layouts write artifacts to the project root.

| Training notebook | Experiments | Final artifact |
|---|---:|---|
| Random_forest_.ipynb | 3 | models/rf_final.joblib |
| Mlp.ipynb | 5 | models/mlp_final.keras |
| CNN.ipynb | 3 | models/cnn_final.keras |

`app.py` is the Streamlit entry point. `assets/` contains the mouse/touch canvas. `preprocessing.py` prepares drawings; `model_io.py` loads and adapts models. `models/` includes selected models and metadata. `results/` contains executed notebooks, tuning histories, metrics, plots, timing and verification evidence. The original assignment brief is preserved.

## Train locally

On the configured Windows laptop:

```powershell
.\tools\train_local.ps1
.\tools\train_local.ps1 -Model mlp
```

The runner starts fresh kernels, requires GPU for the neural networks, and preserves diagnostics on failure. See [local setup](docs/local_setup.md) for another machine. `requirements-training-lock.txt` records the complete working training environment; `requirements.txt` is the minimal app environment. Local GPU packages are separate from Cloud dependencies.

## Methodology and verification

Shared split: 50,000 train / 10,000 validation / 10,000 test, stratified seed 42. Pixels: float32, raw value divided by 255 once. RF/MLP use 784 features; CNN uses 28 by 28 by 1. Select configurations on validation data, then evaluate each selected saved model on the test set. Original team notes/comments are preserved with labelled corrections where needed.

Read the [methodology](docs/methodology.md) and [report](docs/report.md). Run `python -m unittest discover -s tests -v`, `python tools/check_saved_models.py`, or `python tools/compare_results.py`. Evidence is kept beside the measured results. Timing is specific to hardware; a 10,000-image batch is different from one browser prediction.

For optional Windows/Edge browser verification, install `requirements-browser.txt`, start the app, and run `python tools/check_browser.py`. After a new training run, refresh the saved-model and browser checks before running `python tools/build_report.py` to regenerate this README and the report from measured evidence.

Read [what changed and why we trained again](docs/changes.md) for the before/after explanation. **Pushing to GitHub and deploying do not retrain models.** Only the training commands execute the notebooks. Documentation, UI changes and normal app use load the existing saved artifacts.

## GitHub and deployment

Repository: [4awmy/MNIST-Digit-Classifier](https://github.com/4awmy/MNIST-Digit-Classifier). Use separate branches and pull requests for team changes. Backups, datasets, intermediate checkpoints, environments and secrets are excluded from Git; final artifacts are tracked.

The user will deploy `main` / `app.py` with Python 3.12 in Streamlit Community Cloud after the GitHub update. Follow the [deployment guide](docs/deployment.md), verify all three models in the live app, then record its public URL. See [remaining work](docs/next_steps.md).
