# Assignment 1: MNIST Digit Recognition

We compare Random Forest, a fully connected MLP, and a simple CNN, save one selected model from each family, and use all three in a drawing application.

## Data preparation and methodology

MNIST supplies 60,000 development images and a separate 10,000-image test set. Each notebook uses the same stratified split: 50,000 training and 10,000 validation images, seed 42. Pixels are cast to float32 and divided by 255 once. RF and MLP receive 784 features; CNN receives 28 by 28 by 1 images. Integer digit labels are paired with sparse categorical cross-entropy in the networks.

Training fits model parameters; validation selects configurations. The test set is used only after selection. These are small controlled searches with one seed per configuration, so small differences do not establish a statistically reliable improvement.

## Hyperparameter experiments and choices

### Random Forest

| Trial | Trees | Max depth | Validation accuracy | Fit s |
| --- | --- | --- | --- | --- |
| A | 50 | 20 | 96.31% | 2.19 |
| B | 100 | 20 | 96.58% | 4.23 |
| C | 100 | Unlimited | 96.65% | 4.33 |

A to B increases tree count; B to C removes the depth limit. All trials use seed 42 and the same data. Choose highest validation accuracy, breaking ties with fewer trees, then shallower depth and trial name. **C: 100 trees, unlimited depth** wins at 96.65%, a 0.34-point gain over A with increased tree complexity. The final forest is saved with compression.

### Fully connected MLP

| Trial | Hidden units | Learning rate | Batch | L2 | Best epoch | Validation accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| A | [128, 128] | 0.001 | 128 | 0.0 | 7 | 97.27% |
| B | [256, 128] | 0.001 | 128 | 0.0 | 5 | 97.47% |
| C | [128, 128] | 0.0003 | 128 | 0.0 | 17 | 97.39% |
| D | [128, 128] | 0.001 | 512 | 0.0 | 12 | 97.14% |
| E | [128, 128] | 0.001 | 128 | 0.0001 | 6 | 97.41% |

Each alternative changes one baseline setting: width, learning rate, batch size or L2. The architecture uses Dense/ReLU hidden layers and ten softmax outputs. **B: 256/128 hidden units, learning rate 0.001, batch 128, no L2** wins at 97.47%. The wider first layer improves validation accuracy by 0.20 points over A, with a larger parameter count. L2 trial E is close but does not exceed B.

### Simple CNN

| Trial | Filters | Learning rate | Best epoch | Validation accuracy |
| --- | --- | --- | --- | --- |
| A | [16, 32] | 0.001 | 6 | 98.57% |
| B | [32, 64] | 0.001 | 7 | 98.74% |
| C | [16, 32] | 0.0003 | 20 | 98.63% |

Two 3 by 3 convolution/ReLU stages each use 2 by 2 max pooling, followed by flattening, Dense(64, ReLU), and ten softmax outputs. Batch size is 128; no pretrained network is used. **B: 32/64 filters, learning rate 0.001** wins at 98.74%, a 0.17-point gain over A. The lower learning rate in C needs more epochs without exceeding B.

Both networks allow 20 epochs with validation-loss early stopping, patience 3. Each trial saves its minimum-validation-loss checkpoint. Choose among those checkpoints by validation accuracy, then lower validation loss, fewer parameters and trial name. L2 contributes to reported loss, so losses across different penalties are not pure predictive cross-entropy comparisons.

## Evaluation and comparison

The selected saved models are reloaded and evaluated on the 10,000-image test set. Reload consistency is checked using validation predictions.

| Model | Test accuracy | Macro precision | Macro recall | Macro F1 | Size MiB |
| --- | --- | --- | --- | --- | --- |
| Random Forest | 96.95% | 0.9694 | 0.9692 | 0.9693 | 18.94 |
| MLP | 97.62% | 0.9763 | 0.9760 | 0.9761 | 2.72 |
| CNN | 99.02% | 0.9901 | 0.9900 | 0.9901 | 1.43 |

| Model | Selected fit s | Search fit s | 10,000-image inference s | Complexity |
| --- | --- | --- | --- | --- |
| Random Forest | 4.33 | 10.74 | 0.12 | 100 trees / 862,140 nodes |
| MLP | 39.74 | 253.93 | 0.43 | 235,146 parameters |
| CNN | 54.91 | 212.81 | 0.72 | 121,930 parameters |

RF timings used CPU; network training and batch inference used GPU. Timings include checkpoint activity and are hardware-dependent, so they are not an equal-hardware speed comparison. Tree/node counts and neural parameter counts are different measures of complexity. Compressed file size is not runtime memory usage.

## Generalization and errors

RF has a 3.35-point training/validation accuracy gap and 305 test errors. Frequent confusions include 7 as 2 (20 cases), 9 as 3 (16), and 4 as 9 (14). MLP has a 1.41-point gap and 238 errors, including 4 as 9 (21), 5 as 3 (11), and 8 as 3 (9). CNN has a 0.69-point gap and 98 errors, including 0 as 6 (8), 2 as 7 (7), and 4 as 9 (6).

Neural training accuracy is averaged while weights change; its gap from checkpoint validation accuracy is descriptive. Early stopping limits continued fitting after validation loss deteriorates. See the [RF confusion matrix](report_assets/rf_confusion_matrix.png), [MLP confusion matrix](report_assets/mlp_confusion_matrix.png), [CNN confusion matrix](report_assets/cnn_confusion_matrix.png), [MLP learning curves](report_assets/mlp_learning_curves.png), and [CNN learning curves](report_assets/cnn_learning_curves.png).

## Recommended model

**CNN is recommended**, based on the highest validation accuracy, 98.74%. Its independent test accuracy is 99.02%, exceeding MLP by 1.40 points and RF by 2.07 points. Convolutions learn spatial image patterns with shared weights, fitting the digit-recognition task. CNN has fewer parameters than the selected MLP and the smallest saved file, though it has higher training cost. Its smaller accuracy gap and fewer test errors support its generalization on MNIST.

All models remain selectable. Measured warmed CPU prediction medians were 5.06 ms for RF, 1.98 ms for MLP, and 5.20 ms for CNN. These measure model execution after preprocessing, excluding browser/network time. MNIST results do not guarantee accuracy on arbitrary handwriting or photographs.

## Interactive application

The Streamlit app loads the three saved models. A user draws one digit with a mouse or touch, chooses a model, and presses Predict. Clear resets the canvas. The image is converted to grayscale, cropped, resized with its aspect ratio preserved, centered in 28 by 28, and normalized once. RF/MLP flatten it; CNN adds a channel. Blank drawings are rejected, and a preview shows the model input. Editing the drawing or changing models invalidates the previous result. The app performs prediction without training. Run instructions are in [README.md](README.md).
