# Assignment 1: MNIST digit recognition

All three model families were trained locally from fresh kernels. This report uses the corrected runs and saved artifacts, with the original team study notes retained in the notebooks. The notebooks remain separate and feed one shared Streamlit application.

## Data and preprocessing

MNIST provides 60,000 development images and a separate 10,000-image test set. All models use the same stratified 50,000/10,000 training/validation split, with seed 42. Pixels are cast to float32 and divided by 255 once. RF/MLP receive 784 features; CNN receives 28 by 28 by 1 images. Labels remain integer digits; sparse categorical cross-entropy is used by the networks. The test set is scored after configuration selection.

## Hyperparameter experiments

### Random Forest

| Trial | Trees | Max depth | Train accuracy | Validation accuracy | Fit seconds |
| --- | --- | --- | --- | --- | --- |
| A | 50 | 20.0 | 99.94% | 96.31% | 2.19 |
| B | 100 | 20.0 | 99.94% | 96.58% | 4.23 |
| C | 100 | Unlimited | 100.00% | 96.65% | 4.33 |

A to B changes tree count; B to C changes the depth limit. RF uses seed 42 and all CPU cores. Select highest validation accuracy, then fewer trees, shallower depth and trial name. Selected configuration: `{"name": "C", "n_estimators": 100, "max_depth": null}`.

### Fully connected MLP

| Trial | Hidden units | Learning rate | Batch | L2 | Best epoch | Validation accuracy | Fit seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | [128, 128] | 0.001 | 128 | 0.0 | 7 | 97.27% | 49.89 |
| B | [256, 128] | 0.001 | 128 | 0.0 | 5 | 97.47% | 39.74 |
| C | [128, 128] | 0.0003 | 128 | 0.0 | 17 | 97.39% | 95.96 |
| D | [128, 128] | 0.001 | 512 | 0.0 | 12 | 97.14% | 20.79 |
| E | [128, 128] | 0.001 | 128 | 0.0001 | 6 | 97.41% | 47.55 |

Each alternative changes one baseline setting: width, learning rate, batch size, or L2. The network uses Dense/ReLU hidden layers and ten softmax outputs. Training allows 20 epochs with validation-loss early stopping, patience 3. Each trial contributes its minimum-validation-loss checkpoint; select by validation accuracy, then loss, parameter count and trial name. L2 contributes to reported loss, so losses across differing penalties are not pure predictive cross-entropy comparisons. Selected configuration: `{"name": "B", "hidden_units": [256, 128], "learning_rate": 0.001, "batch_size": 128, "l2": 0.0}`.

### Introductory CNN

| Trial | Filters | Learning rate | Batch | Dense units | Best epoch | Validation accuracy | Fit seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | [16, 32] | 0.001 | 128 | 64 | 6 | 98.57% | 50.40 |
| B | [32, 64] | 0.001 | 128 | 64 | 7 | 98.74% | 54.91 |
| C | [16, 32] | 0.0003 | 128 | 64 | 20 | 98.63% | 107.50 |

The CNN uses two 3 by 3 convolution/ReLU stages, each with 2 by 2 max pooling, then flatten, Dense(64, ReLU), and ten softmax outputs. Trials change filter count or learning rate. The same checkpoint, early-stopping and validation-ranking policy applies. No pretrained backbone is used. Selected configuration: `{"name": "B", "filters": [32, 64], "learning_rate": 0.001, "batch_size": 128, "dense_units": 64}`.

## What the experiments show

Validation accuracy changes below are relative to each family's baseline A. They explain the current selection, not a statistically established advantage. Keeping one changed setting per alternative makes the comparison easier to interpret.

| Family | Trial | Validation change vs A (points) | Fit time vs A |
| --- | --- | --- | --- |
| Random Forest | B | +0.27 | 1.93x |
| Random Forest | C | +0.34 | 1.98x |
| MLP | B | +0.20 | 0.80x |
| MLP | C | +0.12 | 1.92x |
| MLP | D | -0.13 | 0.42x |
| MLP | E | +0.14 | 0.95x |
| CNN | B | +0.17 | 1.09x |
| CNN | C | +0.06 | 2.13x |

A wider MLP increases parameter count; lower learning rate changes the optimization pace; larger batches reduce updates per epoch; L2 penalizes large weights. More RF trees increase training/storage cost, and removing the depth cap permits larger trees. Wider CNN filters increase capacity while retaining the same introductory two-stage structure. The selected values are justified by the validation ranking above, with test performance, artifact size and timings providing the final comparison.

## Final saved-model results

| Model | Validation accuracy | Test accuracy | Macro precision | Macro recall | Macro F1 | Size MiB |
| --- | --- | --- | --- | --- | --- | --- |
| Random Forest | 96.65% | 96.95% | 0.9694 | 0.9692 | 0.9693 | 18.94 |
| MLP | 97.47% | 97.62% | 0.9763 | 0.9760 | 0.9761 | 2.72 |
| CNN | 98.74% | 99.02% | 0.9901 | 0.9900 | 0.9901 | 1.43 |

| Model | Selected fit s | Total search fit s | Reload s | 10,000-image inference s | Complexity |
| --- | --- | --- | --- | --- | --- |
| Random Forest | 4.33 | 10.74 | 1.64 | 0.12 | 100 trees; 862,140 nodes |
| MLP | 39.74 | 253.93 | 0.16 | 0.43 | 235,146 parameters |
| CNN | 54.91 | 212.81 | 0.23 | 0.72 | 121,930 parameters |

Hardware: Intel Core i7-13650HX, 20 logical CPUs, 32 GB installed RAM, and NVIDIA RTX 4060 Laptop GPU (8 GB). Ubuntu 24.04/WSL2 used Python 3.12.3, TensorFlow 2.21.0, Keras 3.15.1, and scikit-learn 1.9.1. RF trained/inferred on CPU; MLP/CNN training and notebook batch inference used GPU. Timing includes checkpoint activity and is specific to this laptop. Tree/node counts and network parameter counts are different units.

## Generalization and errors

**Random Forest:** 305 errors out of 10,000. Training accuracy 100.00%, validation accuracy 96.65%, gap 3.35 percentage points. The largest directed confusions were 7 predicted as 2 (20), 9 predicted as 3 (16), 4 predicted as 9 (14).

Evidence: [confusion matrix](../results/rf/confusion_matrix.png), [per-digit metrics](../results/rf/classification_report.csv), [example errors](../results/rf/misclassified_examples.png).

**MLP:** 238 errors out of 10,000. Selected-epoch training accuracy 98.88%, validation accuracy 97.47%, gap 1.41 percentage points. The largest directed confusions were 4 predicted as 9 (21), 5 predicted as 3 (11), 8 predicted as 3 (9).

Evidence: [confusion matrix](../results/mlp/confusion_matrix.png), [per-digit metrics](../results/mlp/classification_report.csv), [example errors](../results/mlp/misclassified_examples.png).

[Training and validation learning curves](../results/mlp/learning_curves.png).

**CNN:** 98 errors out of 10,000. Selected-epoch training accuracy 99.43%, validation accuracy 98.74%, gap 0.69 percentage points. The largest directed confusions were 0 predicted as 6 (8), 2 predicted as 7 (7), 4 predicted as 9 (6).

Evidence: [confusion matrix](../results/cnn/confusion_matrix.png), [per-digit metrics](../results/cnn/classification_report.csv), [example errors](../results/cnn/misclassified_examples.png).

[Training and validation learning curves](../results/cnn/learning_curves.png).

Training-epoch neural accuracy averages predictions made while weights change; validation accuracy uses the checkpoint weights, so this gap is descriptive rather than an exact matched-state measurement. Early stopping limits continued fitting after validation loss deteriorates. These small controlled searches use one seed per setting; small differences require repeated-seed experiments before claiming a reliable improvement.

## Model recommendation and app responsiveness

Recommend **CNN** using validation accuracy (98.74%); its separate test result is 99.02% with macro F1 0.9901. This selection uses validation evidence; the final test table provides independent performance evidence. The CNN uses spatial filters and shared weights suited to handwritten images. Its saved file is 1.43 MiB. All three models remain selectable, making accuracy, complexity and response cost visible.

After selection, the test comparison shows +2.07 percentage points versus Random Forest; +1.40 percentage points versus MLP. The cost is assessed alongside that gain: Random Forest selected fit 4.33s, artifact 18.94 MiB; MLP selected fit 39.74s, artifact 2.72 MiB; CNN selected fit 54.91s, artifact 1.43 MiB. Training times use different CPU/GPU devices, so they describe this workflow rather than equal-hardware speed. The generalization gaps and error patterns above qualify the accuracy comparison.

On the local CPU app check, the recommended model's warmed single-drawing median was 5.20ms. This supports the intended one-digit interaction locally; Cloud responsiveness must still be checked after deployment. Parameter counts, file sizes and measured response times describe different costs and should be considered together.

| Model | CPU load s | Warmed single-drawing median ms | p95 ms |
| --- | --- | --- | --- |
| Random Forest | 1.62 | 5.06 | 6.24 |
| MLP | 0.12 | 1.98 | 2.59 |
| CNN | 0.09 | 5.20 | 6.67 |

The app timings use 100 warmed calls per model on CPU. They measure model execution after preprocessing and exclude browser/network round trips. Ten enlarged MNIST examples were also passed through the complete drawing-preparation pipeline as a smoke check; this is not a handwriting accuracy benchmark. Cloud startup/memory/latency require measurement on the actual deployment.

## Interactive application and verification

Run `python -m streamlit run app.py`. Draw one white digit on the black canvas with a mouse or finger, select CNN/MLP/Random Forest, then press Predict. Clear resets drawing and result. Changing the model keeps the drawing and removes the previous prediction. The preview shows the grayscale image after foreground cropping, aspect-preserving resizing to a 20-pixel bounding box, centering in 28 by 28, center-of-mass adjustment, and float32 normalization. RF/MLP flatten this array; CNN adds a channel. Blank input is rejected. Model scores are not calibrated confidence.

Browser checks passed for blank input, mouse/touch drawing, all three model choices, drawing persistence, stale-result invalidation, Clear, processed preview and comparison display. Evidence and screenshots are in `results/app_browser/`; saved-model checks are in `results/app_smoke_check.json`. Automated preprocessing/comparison checks also pass. Saved-artifact reload consistency was verified by each training notebook.

![Local drawing application](../results/app_browser/desktop.png)

## Reproducibility, collaboration and limitations

The public repository is [4awmy/MNIST-Digit-Classifier](https://github.com/4awmy/MNIST-Digit-Classifier). The team has sent its collaborator invitation. The notebooks, source code, final artifacts, metrics, environment records and dependency locks support review and reproduction; backups and intermediate checkpoints are excluded from Git. Team names are omitted as requested.

See [methodology](methodology.md), [local setup](local_setup.md), [what changed and why we retrained](changes.md), and [deployment guide](deployment.md). The user will deploy after the GitHub update. Record the final Cloud URL only after a successful live deployment and prediction check. Local success does not establish Cloud success. Pushing and deployment load the saved artifacts; they do not execute training.

MNIST is a clean centered single-digit dataset. Arbitrary drawing styles and photographs can differ substantially; this app does not segment multiple digits. Repeated seeds, broader validation-driven searches and a labelled drawing evaluation set would strengthen the evidence. Picture upload remains the next optional feature after deployment and final documentation review.
