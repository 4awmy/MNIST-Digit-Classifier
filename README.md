# MNIST Digit Classifier

Assignment 1 for Professional Training in AI II: train and compare a Random Forest, a fully connected multilayer perceptron (MLP), and a simple convolutional neural network (CNN), then recognize a digit drawn in a Streamlit application.

**[Open the Streamlit app](https://mnist-digit-classifier-aast.streamlit.app/)** · **[Read the assignment report](report.md)**

The app supports mouse and touch drawing, model selection, clearing the canvas, a processed-image preview, and scores for digits 0–9. All three trained models are included. **CNN is the recommended default:** it achieves **99.02% test accuracy** on MNIST.

## Contents

- [Use the application](#use-the-application)
- [Run the project](#run-the-project)
- [Files and assignment deliverables](#files-and-assignment-deliverables)
- [How the system works](#how-the-system-works)
- [Data and preprocessing](#data-and-preprocessing)
- [The three models](#the-three-models)
- [Training, tuning, and evaluation](#training-tuning-and-evaluation)
- [Results and trade-offs](#results-and-trade-offs)
- [Functions and component behavior](#functions-and-component-behavior)
- [Imports and dependencies](#imports-and-dependencies)
- [Saved models, GitHub, and deployment](#saved-models-github-and-deployment)
- [Limitations and troubleshooting](#limitations-and-troubleshooting)

## Use the application

1. Open the [app](https://mnist-digit-classifier-aast.streamlit.app/).
2. Select **CNN**, **MLP**, or **Random Forest**. CNN is selected initially.
3. Draw one large digit from **0 to 9** on the black canvas using a mouse, finger, or pen. Use one digit per prediction.
4. Press **Predict**. The result shows the recognized digit, model name, and model execution time.
5. Expand **What the model sees** to inspect the centered 28×28 input, or **Scores for all digits** to compare the ten output scores.
6. Press **Clear** to start another drawing. After editing a drawing or changing models, press **Predict** again; the previous result is cleared to avoid showing stale scores.

**Compare the three models** displays the saved MNIST test results. That table describes the benchmark, not the correctness of the current drawing. The score chart contains normalized model scores; a high score is not a guaranteed probability of being correct.

## Run the project

Use **Python 3.12**. From the project folder, create an environment and install the pinned dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

On macOS/Linux, use `source .venv/bin/activate` for the activation step. Open **http://localhost:8501** after Streamlit starts. Run commands from the folder containing `app.py` and `requirements.txt`.

The app requires the three existing files in `models/`. **Starting the app does not train models or download MNIST.** The requirements use CPU TensorFlow for application inference; a GPU is not required to recognize a drawing.

To open the training notebooks, install Jupyter separately in the same environment:

```powershell
python -m pip install jupyterlab
python -m jupyterlab
```

Open a notebook from the project root and run its cells in order. Select the Python environment in which the requirements were installed. Training can use CPU or a separately configured compatible GPU environment; the supplied requirements do not install a GPU setup.

## Files and assignment deliverables

| File or folder | Purpose |
| --- | --- |
| [Random_forest_.ipynb](Random_forest_.ipynb) | Load and prepare MNIST, tune the forest, evaluate and save the selected forest. |
| [Mlp.ipynb](Mlp.ipynb) | Train and tune the Dense-only neural network; save the selected checkpoint and figures. |
| [CNN.ipynb](CNN.ipynb) | Train and tune the introductory CNN; save the selected checkpoint and figures. |
| [models/](models/) | `rf_final.joblib`, `mlp_final.keras`, and `cnn_final.keras`: the three saved models used by the app. |
| [report.md](report.md) | Short report: preprocessing, hyperparameter experiments, comparison, generalization, costs, and model selection. |
| [report_assets/](report_assets/) | Three test confusion matrices and two sets of neural learning curves, embedded and interpreted in the report. |
| [app.py](app.py) | Streamlit interface, model caching, session state, prediction display, and comparison table. |
| [preprocessing.py](preprocessing.py) | Decode a drawing, prepare its pixels, and reshape the input for each model. |
| [model_io.py](model_io.py) | Load saved artifacts, check their interfaces, and produce a common prediction result. |
| [canvas_component.py](canvas_component.py) | Register the custom Streamlit drawing component. |
| [assets/](assets/) | Canvas HTML, CSS, and JavaScript for drawing, Clear, and Predict. |
| [results/comparison.csv](results/comparison.csv) | One summary row per selected model; supplies the app's comparison table. |
| [requirements.txt](requirements.txt) | Pinned application and notebook dependencies. |
| [.streamlit/config.toml](.streamlit/config.toml) | Shared theme, headless server setting, and usage-statistics setting. |
| [.gitignore](.gitignore) | Exclude environments, caches, temporary checkpoints, backups, and secrets from Git. |

These map to the five assignment deliverables: training source, saved models, a short report, application source, and run instructions. The three notebooks stay **separate** because each contains one model family's complete experiment. They share the same dataset split and evaluation rules; the app and report combine their results. Combining them into one notebook is unnecessary.

## How the system works

There are two workflows. Training creates the artifacts; the application loads them for inference.

```text
TRAINING — run notebook cells deliberately
MNIST → shared train/validation/test split → preprocessing
      → hyperparameter trials → validation selection → saved model
      → selected-model test evaluation → figures + comparison.csv

APPLICATION — each Predict click
Browser canvas → PNG snapshot → decode_canvas → prepare_digit
               → model_input → selected saved model
               → predicted digit + ten scores + execution time
```

The notebooks learn model parameters from training data. Hyperparameters are the settings chosen before fitting, such as tree count or learning rate. The application only applies those learned parameters to a new image. `results/comparison.csv` is display data, not model weights; a missing comparison table does not prevent prediction when the model files are available.

## Data and preprocessing

### MNIST used for experiments

MNIST consists of grayscale images of single handwritten digits, each **28×28 pixels**, with labels **0–9**. Keras supplies 60,000 development images and a separate 10,000-image test set. Each notebook uses `train_test_split` with `test_size=10000`, `random_state=42`, and stratification by label:

| Partition | Images | Role |
| --- | --- | --- |
| Training | 50,000 | Fit tree decisions or neural weights. |
| Validation | 10,000 | Select hyperparameters and neural checkpoints. |
| Test | 10,000 | Evaluate the selected model from each family. |

Stratification approximately preserves the proportion of every digit in both development partitions. The same seed and split across notebooks support a fair accuracy comparison. Test images are not used to select hyperparameters or stopping epochs.

Pixel arrays are converted to `float32` and divided by **255 exactly once**, converting intensities from 0–255 to 0–1. Normalization helps neural optimization and gives all models a consistent input contract; trees do not require scaling in the same way. No additional standardization, augmentation, or external training images are used.

| Model | One training example | One application batch |
| --- | --- | --- |
| Random Forest | `(784,)` flattened pixels | `(1, 784)` |
| MLP | `(784,)` flattened pixels | `(1, 784)` |
| CNN | `(28, 28, 1)` image with grayscale channel | `(1, 28, 28, 1)` |

Flattening preserves pixel order but removes the explicit two-dimensional input structure. Integer labels are retained. The networks use **sparse categorical cross-entropy**, which accepts these integers; one-hot labels with categorical cross-entropy would be an alternative representation, not a required extra step.

### Drawings used for prediction

The browser canvas is **280×280**, with white strokes on black, matching MNIST's bright-digit/dark-background convention. It is prepared differently from the already formatted MNIST images:

1. Decode the PNG data URL. Reject malformed input, encoded payloads over 3,000,000 characters, or images over 1,000,000 pixels.
2. Composite transparency onto black and convert to grayscale.
3. Find foreground pixels brighter than **15**. Fewer than **10** foreground pixels is treated as a blank drawing.
4. Crop to the foreground bounding box. This threshold locates the bounds; it does **not** turn the image into a binary black-and-white mask.
5. Resize with LANCZOS interpolation, preserving aspect ratio, so the longest side is **20 pixels**.
6. Place the resized digit in a black **28×28** square, then shift it toward the center using its intensity-weighted center of mass.
7. Convert to `float32`, divide by 255 once, and reshape for the selected model.

Cropping removes unused canvas space, aspect-ratio preservation avoids stretching, and centering reduces position differences from MNIST. The preview makes this transformation visible. These are practical heuristics; unusual strokes or drawings at the canvas edge can still be difficult to recognize. `model_input` checks shape, finite values, and the 0–1 range without dividing a second time.

## The three models

### Random Forest: classical baseline

The selected `RandomForestClassifier` uses **100 trees**, **unlimited maximum depth**, and **seed 42**. Trees learn decisions on individual pixel features using randomized training samples and feature subsets. The classifier averages the trees' class-probability estimates and selects the largest score.

The experiment compares 50 versus 100 trees, then capped versus unlimited depth. More trees can make the ensemble more stable at increased compute and storage cost; removing the depth cap allows more detailed decisions but can increase overfitting. Trial C has the best validation accuracy. The selected forest contains **862,140 nodes**, with observed maximum depth **41**. It is saved using compressed Joblib serialization.

RF is a useful non-neural baseline and fits quickly in the measured experiment. It does not explicitly model neighboring pixels as image features, and its saved file is much larger than either network.

### MLP: fully connected neural network

The selected architecture is:

```text
784 pixels → Dense(256, ReLU) → Dense(128, ReLU) → Dense(10, softmax)
```

It uses **235,146 trainable parameters**, Adam with **learning rate 0.001**, **batch size 128**, and **no L2 penalty** in the selected trial. ReLU applies `max(0, x)` and provides nonlinear hidden transformations. Softmax produces ten scores summing to one. A Dense layer with `a` inputs and `b` outputs has `a × b + b` parameters, including biases.

The five trials vary first-layer width, learning rate, batch size, or L2 relative to a 128/128 baseline. Increasing width to 256/128 gives the highest validation accuracy. A smaller learning rate needs more epochs; a larger batch changes the number of updates per epoch; L2 discourages large weights but does not win this search. The active architecture uses fully connected layers only, with no Dropout layer.

The MLP is compact and has the fastest measured warmed single-image CPU prediction. Its Dense layers learn from the fixed pixel vector without the CNN's explicit local receptive fields or shared convolutional weights.

### CNN: simple image model

The selected network has two convolution/pooling stages and a small classifier:

| Stage | Output shape, excluding batch | Parameters |
| --- | --- | --- |
| Input | 28×28×1 | 0 |
| Conv2D, 32 filters, 3×3, ReLU | 26×26×32 | 320 |
| Max pooling, 2×2 | 13×13×32 | 0 |
| Conv2D, 64 filters, 3×3, ReLU | 11×11×64 | 18,496 |
| Max pooling, 2×2 | 5×5×64 | 0 |
| Flatten | 1,600 | 0 |
| Dense, 64, ReLU | 64 | 102,464 |
| Dense, 10, softmax | 10 | 650 |
| **Total** | | **121,930** |

Convolutions use stride 1 and valid padding. Shared filters detect local patterns across positions; pooling reduces spatial dimensions. This is an introductory CNN with no pretrained backbone. Adam uses **learning rate 0.001**, **batch size 128**, and **64 Dense hidden units**. Three trials compare 16/32 versus 32/64 filters and a lower learning rate. The wider filter configuration wins on validation accuracy.

## Training, tuning, and evaluation

Each notebook follows the same order: load data, create the split, normalize and reshape, define a model builder, run trials, select a winner using validation results, save/reload it, evaluate the test set, and export figures and the comparison row. The [report](report.md#hyperparameter-search-and-final-choices) contains every trial and its measured validation result.

**Forest selection:** highest validation accuracy; ties prefer fewer trees, then shallower maximum depth, then trial name. `n_jobs=-1` allows parallel training. The app changes the loaded forest to `n_jobs=1` to avoid parallel scheduling overhead for a single drawing.

**Network selection has two levels:**

1. Within each trial, save the checkpoint with **minimum validation loss** using `ModelCheckpoint`. `EarlyStopping` monitors that same loss with **patience 3**, restores the best weights, and training has a maximum of **20 epochs**. `TerminateOnNaN` stops invalid numerical training.
2. Reload each trial's saved checkpoint and rank those checkpoints by **highest validation accuracy**, then lower validation loss, fewer parameters, and trial name.

The checkpoint is not necessarily the final epoch or the epoch with maximum accuracy: loss considers the scores assigned to the correct labels, not just whether the largest score is correct. The selected MLP checkpoint is epoch **5**; the selected CNN checkpoint is epoch **7**. Final models are the selected trained artifacts, not an additional fit on the combined training/validation data.

Seeds are reset for neural trials and TensorFlow deterministic operations are requested. This improves repeatability; different platforms or numerical libraries can still produce different timings or results. The search is a small controlled comparison with one seed per configuration, not an exhaustive search or repeated-seed confidence study.

**Evaluation metrics:** accuracy is the fraction of correct digits. Per-digit precision measures how often predictions of that digit are correct; recall measures how many examples of that true digit are found. F1 is their harmonic mean. **Macro** precision, recall, and F1 average the ten per-digit values equally. Confusion matrices use true digits as rows and predicted digits as columns. Neural learning curves show training and validation loss/accuracy over epochs.

To rerun experiments, execute the relevant notebook from top to bottom. It overwrites that family's final model and report figures, and replaces its row in `results/comparison.csv` while preserving the others. Temporary neural checkpoints are written under `.checkpoints/` and ignored by Git. Run notebooks one at a time when updating shared output files. The recorded Markdown summaries describe the included artifacts; if training is rerun, update those summaries and `report.md` to match the new measurements. A documentation edit or Git push is not a reason to retrain.

## Results and trade-offs

| Model | Selected trial | Validation accuracy | Test accuracy | Macro F1 | Saved size |
| --- | --- | --- | --- | --- | --- |
| Random Forest | C | 96.65% | 96.95% | 0.9693 | 18.94 MiB |
| MLP | B | 97.47% | 97.62% | 0.9761 | 2.72 MiB |
| CNN | B | **98.74%** | **99.02%** | **0.9901** | **1.43 MiB** |

CNN is selected because it has the highest validation accuracy and its spatial architecture suits image recognition. Independent test evaluation supports that choice: it makes **98 errors**, versus **238** for MLP and **305** for RF, on 10,000 images. Its test accuracy exceeds MLP by **1.40 percentage points** and RF by **2.07 points**. It also has fewer parameters than the MLP and the smallest saved artifact.

| Consideration | Random Forest | MLP | CNN |
| --- | --- | --- | --- |
| Selected-trial fit time | 4.33 s | 39.74 s | 54.91 s |
| Total trial fit time | 10.74 s / 3 trials | 253.93 s / 5 trials | 212.81 s / 3 trials |
| Warmed CPU single-image median | 5.06 ms | **1.98 ms** | 5.20 ms |
| Main strength | Fast measured fitting; classical baseline | Fast measured single-image inference; simple Dense architecture | Highest accuracy; spatial feature sharing; smallest file |
| Main cost | Large forest artifact; lower accuracy | More parameters than CNN; no explicit image structure | More convolution computation; longer selected fit than MLP |

Fit timings used CPU for RF and GPU for networks, and neural fit timings include checkpoint activity. They are **not an equal-hardware benchmark**. Search totals also reflect different trial counts and stopping times. Single-image medians used 100 warmed CPU model calls per model and exclude preprocessing, browser rendering, and network time. Disk size is not memory usage, and tree nodes are not directly comparable to neural parameters. Read the [report](report.md) for full metrics, batch timings, error patterns, and figures.

The app keeps all three models selectable so a user can compare behavior on the same drawing. Preloading and caching all three makes switching straightforward, at the cost of startup time and memory. The custom canvas avoids a separate drawing-package dependency but requires maintaining its JavaScript and event handling. CPU inference simplifies the application environment; it does not reproduce GPU training timings.

## Functions and component behavior

### Python application functions

| Function or object | Responsibility and output |
| --- | --- |
| `DrawingError` in `preprocessing.py` | A `ValueError` subclass for input problems that can be explained to the user. |
| `decode_canvas(data_url)` | Validate and decode the canvas PNG data URL; return a loaded Pillow image or raise `DrawingError`. |
| `prepare_digit(image)` | Crop, resize, center, and normalize a Pillow image; return a `float32` array of shape `(28, 28)`. |
| `model_input(digit, model_name)` | Validate the normalized image and return a batch with the model's expected shape. |
| `MODEL_FILES` in `model_io.py` | Map the three displayed names to their saved filenames. |
| `load_artifact(name, path)` | Load Joblib or Keras artifacts and check their input/output contract. RF must have 784 features and classes 0–9; networks must accept the expected shape and output ten scores. |
| `predict_digit(model, name, digit)` | Adapt input, execute inference, validate and normalize scores, and return `digit`, `scores`, `milliseconds`, and `model`. |
| `invalidate_prediction()` in `app.py` | Remove the previous prediction, processed preview, and error from the user's session when input changes. |
| `cached_model(name, path, modified_ns)` | Use `st.cache_resource` to reuse a loaded model. File modification time participates in the cache key so a replaced artifact can be reloaded. |
| `digit_canvas` in `canvas_component.py` | Registered Streamlit v2 component connecting the three browser assets to Python. |

Neural artifacts are loaded with `compile=False` because prediction does not need an optimizer or training loss. Inference uses `training=False`. RF probability columns are mapped through `classes_` so scores correspond to digits 0–9. The reported milliseconds measure the model execution section and score-array conversion, not PNG decoding, preprocessing, final score validation, or complete request latency.

`st.session_state` retains each user's drawing, result, preview, and input error across reruns. `st.cache_resource` retains loaded model resources across reruns and sessions. A missing or incompatible artifact produces a loading message; prediction is disabled for an unavailable selection. The app does not silently train a replacement model.

### Notebook functions

| Function | Purpose |
| --- | --- |
| `build_rf(n_estimators, max_depth)` | Construct a seeded forest with the requested size and depth. |
| `run_rf(config)` | Fit a trial, measure training/validation accuracy and fit time, record tree complexity, and retain the trial model. |
| `build_mlp(hidden_units, learning_rate, l2)` | Build and compile the requested Dense/ReLU network with optional hidden-weight L2. |
| `run_mlp(config)` | Reset state/seed, fit with callbacks, reload the best-loss checkpoint, and return its metrics and settings. |
| `build_cnn(filters, learning_rate, dense_units)` | Build and compile the two-stage CNN with the requested filter counts and classifier width. |
| `run_cnn(config)` | Fit and verify a CNN checkpoint, record its metrics and settings, and retain the learning history. |

The neural `histories` dictionaries supply learning curves; RF's `trial_models` holds candidates until selection. `shutil` copies the chosen neural checkpoint to its final filename so the deployed artifact is the checkpoint that was evaluated.

### Browser canvas

`canvas.html` defines the canvas and buttons; `canvas.css` provides responsive styling and disables touch scrolling over the drawing area. `canvas.js` exports the component handler:

- `black()` resets the bitmap to black; `point(event)` converts screen coordinates to canvas coordinates.
- Pointer down/move/up handlers draw rounded white strokes and support mouse, touch, and pen. Pointer capture keeps strokes connected while a pointer moves beyond the canvas boundary.
- `commit()` saves the drawing as a PNG through `setStateValue('drawing', ...)` after a stroke.
- **Predict** snapshots the currently visible bitmap directly, sends it through `setTriggerValue('predict', ...)`, and includes a timestamp so repeated clicks trigger new predictions.
- **Clear** resets the bitmap and drawing state and sends a clear event. The component restores the saved bitmap after Streamlit reruns and removes old handlers during cleanup.

## Imports and dependencies

| Dependency / import | Why it is used |
| --- | --- |
| `tensorflow`, `tensorflow.keras`, `layers`, `regularizers` | Load MNIST; build/train neural networks; Adam, losses, callbacks, seeds, deterministic operations, and saved-model loading. L2 is used only in its experimental MLP trial. |
| `keras` | Keras API and `.keras` serialization support used through TensorFlow's Keras interface. |
| `sklearn.model_selection.train_test_split` | Shared stratified training/validation split. |
| `sklearn.ensemble.RandomForestClassifier` | Classical model training and class-probability prediction. |
| `sklearn.metrics` | Accuracy, classification reports, macro precision/recall/F1, and confusion matrices. |
| `numpy as np` | Pixel arrays, normalization, reshaping, finite-value checks, score handling, and numerical calculations. |
| `pandas as pd` | Trial/comparison tables, CSV reading/writing, and the app's score-chart data. |
| `matplotlib.pyplot as plt` | Notebook images, error examples, learning curves, and saved report figures. |
| `PIL.Image` from Pillow | PNG loading, grayscale conversion, compositing, resizing, and centering. |
| `joblib` | Save and load the compressed scikit-learn forest. |
| `streamlit as st` | Web interface, custom component, session state, resource cache, and result displays. |
| `IPython.get_ipython` | Enable inline Matplotlib figures when a notebook kernel is available. |
| `h5py` | Underlying support for Keras weight storage; not directly imported by project modules. |

Standard-library imports need no separate installation: `pathlib.Path` resolves portable project-relative paths; `time.perf_counter` measures elapsed time; `shutil` copies selected checkpoints; `base64`, `binascii`, and `io` decode canvas data; `os` sets default CPU visibility and TensorFlow log verbosity for the app. Exact package versions are in [requirements.txt](requirements.txt). JupyterLab is an optional notebook editor, installed separately.

## Saved models, GitHub, and deployment

The repository includes the trained artifacts needed for prediction. `.keras` is used for the networks and `.joblib` for the forest. Keep these files together with the source and dependencies. Replacing a model must preserve the preprocessing and input/output contract or the loader will reject it.

**Committing, pushing, restarting, or deploying does not retrain anything.** A deployment installs dependencies, executes `app.py`, and loads saved weights. Training occurs only when the notebook training cells are deliberately executed. Retrain when changing the data, model design, or training hyperparameters; ordinary README, report, or interface text updates do not require it.

The application URL is [mnist-digit-classifier-aast.streamlit.app](https://mnist-digit-classifier-aast.streamlit.app/). The project deployment configuration uses this repository, branch **`main`**, entry point **`app.py`**, and Python **3.12**. No separate training service is called by the application. Keep environments, dataset caches, temporary checkpoints, and credentials out of Git, as configured in `.gitignore`.

## Limitations and troubleshooting

MNIST accuracy measures performance on standardized digit images. It does not establish the same accuracy on freehand canvas drawings or photographs. Letters, multiple digits, and arbitrary shapes are outside the task; the classifier still chooses one of ten classes and has no out-of-distribution detector. Image upload is not currently implemented. A future upload feature needs its own background/polarity handling and evaluation rather than assuming every photograph matches the canvas format.

| Symptom | Check or action |
| --- | --- |
| Blank-drawing message | Draw one visible digit, then press Predict. Tiny marks may fail the foreground check. |
| Wrong digit | Inspect **What the model sees**, draw larger with a recognizable shape, or compare another model. A prediction is not guaranteed. |
| Result disappears after editing or switching model | Expected behavior: the previous result is invalidated. Press Predict again. |
| Missing/failed model message | Ensure all three artifacts exist under `models/`; use the pinned package versions and inspect **Loading details**. |
| Package import failure | Install `requirements.txt` in the environment used by Streamlit or the selected notebook kernel. |
| Notebook cannot load MNIST on first run | Keras needs access to download the dataset initially; later runs can use its cache. |
| New training results differ from the report | Update the final artifact, comparison row, figures, notebook summary, and report together; record the actual run's measurements. |

Further work could evaluate a labeled set of canvas drawings, repeat experiments across seeds, test a wider validation-only search, and add image upload after the current documentation and drawing workflow are complete.
