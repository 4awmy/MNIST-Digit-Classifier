# Methodology

## How the project fits together

The project has three independent training notebooks and one prediction application. Each notebook trains one model family and saves its chosen final artifact. The app loads those three artifacts and routes a drawing to the selected model. There is no ensemble and no need to merge the notebooks.

```text
Random_forest_.ipynb → models/rf_final.joblib ┐
Mlp.ipynb           → models/mlp_final.keras ├→ Streamlit drawing app
CNN.ipynb           → models/cnn_final.keras ┘
```

## Dataset and split

MNIST provides 60,000 development images and a separate 10,000-image test set. The development data is split into 50,000 training and 10,000 validation images with `test_size=10000`, `random_state=42`, and `stratify=y_train_full`. All notebooks make the identical split from identical source arrays.

Training data fits model parameters. Validation data selects configurations and, for the networks, checkpoints. Test data evaluates only the selected final saved model. Preparing the test array is allowed before selection; scoring it to choose hyperparameters is not.

## Image and label preparation

Cast raw 0–255 pixels to float32 and divide by 255 once. MLP/Random Forest flatten images into 784 features; CNN retains the spatial grid and adds a grayscale channel, giving 28×28×1.

Labels remain integer digits 0–9. The neural networks use sparse categorical cross-entropy, which correctly accepts integer classes. One-hot labels with categorical cross-entropy are an alternative format, not a requirement for classification. Pixel scaling is not essential for decision trees but keeps all three input contracts consistent.

## Experiments

| Family | Configurations tested | Controlled changes |
|---|---:|---|
| Random Forest | 3 | 50→100 trees at depth 20; then unlimited depth with 100 trees |
| MLP | 5 | Baseline 128/128; wider first layer; lower learning rate; larger batch; L2 regularization |
| CNN | 3 | Baseline 16/32 filters; 32/64 filters; lower learning rate with baseline filters |

The MLP uses only Dense computational layers with ReLU hidden activations and ten softmax outputs. The introductory CNN uses two convolution/pooling stages, a small Dense hidden layer, and ten softmax outputs; no pretrained backbone is used.

Each neural-network trial starts with new weights, seed 42, maximum 20 epochs, and validation-loss early stopping with patience 3. Save only the minimum-validation-loss checkpoint for each trial. Choose highest validation accuracy among those saved checkpoints, breaking ties with lower validation loss, fewer parameters, then trial name. L2 contributes to the reported loss, so cross-trial losses with different penalties are not pure predictive cross-entropy.

RF uses seed 42 and all available CPU cores during training. Choose highest validation accuracy; break ties with fewer trees, shallower depth limits, then trial name. Its tree/node counts measure complexity differently from network parameter counts.

These are small controlled searches, not exhaustive grids. One seed per configuration limits claims about tiny differences.

## Local execution and reproducibility

Train locally under Ubuntu 24.04/WSL2 in an isolated Python 3.12 environment. Random Forest uses CPU; MLP/CNN use the RTX 4060. Verify an actual GPU computation before network training, enable GPU memory growth, and run each notebook in a fresh kernel. Execute the three families sequentially to limit memory use.

The runner saves executed notebooks, live logs, and success/failure status. Failed runs keep diagnostic outputs; a failed model is not presented as complete. Notebook metadata records split, configuration, versions, device information, and metrics. Working dependency versions are locked after installation checks.

Original team notes are retained verbatim. Clearly labelled corrections explain outdated statements; comments attached to changed code are retained in historical study excerpts. Current executable code determines the actual experiment.

## Evaluation and final selection

Reload the selected artifact before test evaluation. Export accuracy, macro precision/recall/F1, per-digit precision/recall/F1, a confusion matrix, and example errors. Verify prediction consistency across saving/reloading.

Record selected-fit time, total search fit time, loading time, warmed 10,000-image inference time, and artifact size. Fit timing includes the notebook's checkpoint activity. Compare timing with hardware context: CPU RF and GPU networks do not use identical devices. Batched inference time is not single-drawing response time.

Use learning curves for MLP/CNN and training/validation accuracy gaps for RF to discuss generalization. Choose the project recommendation using predictive performance, errors, complexity, memory/file size, training cost, and measured app responsiveness. All three models remain available in the app regardless of the recommendation.

## Drawing-to-prediction pipeline

1. Draw one bright digit on the app's dark 280×280 canvas using mouse or touch.
2. Select a model and click Predict; the component captures the current canvas as PNG.
3. Convert to grayscale, composite transparency on black, reject blank input, and locate foreground pixels.
4. Crop to content, preserve the aspect ratio while resizing its longest dimension to 20 pixels, and place it in a 28×28 dark image.
5. Center the intensity-weighted center of mass and normalize to float32 [0,1]. Show this processed image in the UI.
6. Route shape `(1,784)` to RF/MLP or `(1,28,28,1)` to CNN. Display the highest-scoring digit.

Clear removes the drawing and prior result. Changing the model preserves the drawing and invalidates the prior prediction. Model scores are not guaranteed calibrated confidence. The application loads saved models through caching and performs inference only; training is separate.

MNIST test performance does not establish reliability on arbitrary drawings or photographs. Picture upload is a later extension after the required drawing workflow and documentation.
