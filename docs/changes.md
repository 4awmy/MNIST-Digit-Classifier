# What changed and why we trained again

## Why retrain?

Training again creates saved models and measured results that correspond to the revised source code. Old notebook outputs do not prove that the revised code runs or that the application can reload its artifacts.

The original notebooks already contained useful work. RF and CNN already used a stratified validation split and tried several configurations. The original MLP's one-hot labels with categorical cross-entropy were also correct. The changes below extend that work and make the three experiments consistent; they do not imply that every original implementation was wrong.

## Changes by notebook

| Area | Original work | Revised implementation |
|---|---|---|
| MLP data | Used the full 60,000 training images; normalized without an explicit float32 cast; converted labels to one-hot | Uses the same stratified 50,000/10,000 training/validation split as RF/CNN; float32 normalization; integer labels with sparse cross-entropy |
| MLP architecture | Two Dense(128) layers, Dropout(0.25), ten softmax outputs | Dense hidden layers and softmax output; removed Dropout to follow the assignment's literal "fully connected layers only" wording. Dropout itself was not a mathematical error |
| MLP experiments | One Adam configuration, batch 512, ten epochs, no validation monitoring | Five controlled trials varying width, learning rate, batch size or L2; up to twenty epochs; validation-loss checkpoints and early stopping |
| MLP reproducibility and saving | NumPy seed only; no final save/reload step | Keras seed 42, deterministic TensorFlow operations, selected `.keras` artifact, reload verification and environment records |
| RF | Three tree/depth experiments, validation comparison, compression and final save already present; selected C explicitly | Keeps those three experiments, selects the winner programmatically from validation results, stores one compressed final artifact and metadata, and adds consistent metrics and plots |
| CNN | Basic two-stage convolution architecture and three experiments already present; initial checkpoint omitted `save_best_only`; final selection explicitly loaded B | Keeps the basic architecture/search, uses best-only checkpoints consistently, adds early stopping, and selects the checkpoint programmatically from current validation results |
| All three | Different reporting/export conventions; Colab-oriented download code in RF | Project-relative `models/` and `results/`; locally executable notebooks; tuning CSVs, test accuracy, macro precision/recall/F1, confusion matrices, example errors, artifact size and timing |

Switching from one-hot labels plus categorical cross-entropy to integer labels plus sparse categorical cross-entropy is an equivalent label representation for this task. The labels and loss must match. This switch alone does not establish a performance improvement.

Neural checkpoints are selected at minimum validation loss within each trial. Trial winners are chosen by validation accuracy, with documented tie breakers. The test set is used to report each selected model's performance after tuning, not to choose hyperparameters. These are small searches with one seed per trial, not exhaustive optimization.

## Application and project additions

- One Streamlit application loads all three saved models. The three training notebooks remain separate.
- A mouse/touch canvas supplies a single white digit on a black background, with Predict, Clear and model selection.
- Shared drawing preparation crops the foreground, preserves aspect ratio, centers a 20-pixel bounding box inside 28 by 28, adjusts the center of mass and normalizes once. A preview shows the resulting input.
- Model adapters flatten RF/MLP inputs and add the CNN channel dimension. Blank drawings are rejected; editing or switching models clears stale predictions while preserving the drawing.
- Added methodology, the evidence-based report, run/deployment instructions, dependency pins and locks, local training helpers, automated checks and browser evidence.
- Original team notes and comments are retained. Historical code snippets are labelled, so they are not mistaken for the active implementation. Local backups are excluded from Git.
- RF training uses CPU; MLP/CNN training uses the laptop's RTX 4060 through WSL2. The inference app uses CPU dependencies suitable for the intended deployment.

## Does pushing or deploying retrain?

**No.** A GitHub push transfers committed files. Streamlit installs the application dependencies and runs `app.py`, which loads `models/rf_final.joblib`, `models/mlp_final.keras` and `models/cnn_final.keras`. It does not execute the training notebooks. This project has no automatic training workflow.

| Change | Retrain? |
|---|---|
| Push files to GitHub, invite a collaborator, or deploy/restart the app | No |
| Edit documentation, styling, buttons or layout | No |
| Add picture upload using the existing drawing preprocessing and saved models | Usually no; test its input preparation and predictions |
| Change training data, architecture, learning rate, batch size, regularization or epoch policy | Yes, rerun the affected experiments and regenerate their saved models/results |
| Change only drawing crop/centering or other app preprocessing | First validate on representative drawings; retrain only if the model input/training contract also needs to change |
| Change library versions | Check artifact compatibility; do not retrain automatically just to push |

To deliberately retrain on the configured laptop, run `.\tools\train_local.ps1` for all models, or `.\tools\train_local.ps1 -Model mlp` for MLP. Regenerate comparison/report evidence after affected runs. Training overwrites the selected artifacts; reviewing documentation never starts training.

## Next handoff

The user will deploy after the GitHub update. Follow [deployment](deployment.md), test the live app with every model and record the live URL. Picture upload is a later extension after the core documentation is reviewed.
