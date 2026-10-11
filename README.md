# MNIST Digit Classifier

Assignment 1: train Random Forest, MLP and CNN models, compare them, and predict digits drawn in a Streamlit app.

## Run the application

Use Python 3.12. From the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

On macOS/Linux, activate with `source .venv/bin/activate` instead. Open **http://localhost:8501**, draw one digit from 0 to 9 with your mouse or finger, choose a model, and click **Predict**. **Clear** resets the drawing. The preview shows the prepared input.

For Streamlit Community Cloud, select this repository, branch `main`, entry point `app.py`, and Python 3.12. The app loads the committed models; pushing or deploying does not train them.

## Assignment deliverables

| Deliverable | Files |
|---|---|
| Training source | Random_forest_.ipynb, Mlp.ipynb, CNN.ipynb |
| Three saved models | models/rf_final.joblib, models/mlp_final.keras, models/cnn_final.keras |
| Short report and figures | report.md, report_assets/ |
| Drawing app source | app.py, model_io.py, preprocessing.py, canvas_component.py, assets/ |
| Run instructions and dependencies | README.md, requirements.txt |

`.streamlit/config.toml` sets the app theme. `results/comparison.csv` supplies the app's comparison table. `.gitignore` excludes temporary files and checkpoints. These support the required deliverables.

## Results

| Model | Validation accuracy | Test accuracy |
| --- | --- | --- |
| Random Forest | 96.65% | 96.95% |
| MLP | 97.47% | 97.62% |
| CNN | 98.74% | 99.02% |

**Recommended model: CNN.** Read [the report](report.md) for preprocessing, hyperparameter choices, evaluation, costs and the model-selection justification.

## Training notebooks

Keep the three notebooks separate. Open them in Jupyter with the project folder as the working directory and run the cells in order. If Jupyter is not already installed, run `python -m pip install jupyterlab`, then `python -m jupyterlab`.

Training can use CPU or a compatible GPU. Each notebook tunes on validation data, saves its selected model, checks reloading and updates the comparison data. Network checkpoints are temporary and excluded from Git. Training is only started by running the notebook training cells; the drawing app does not execute them.

The recorded result summaries and report describe the included saved models. If you deliberately rerun training, update those summaries and the report with the new measurements. Pixels must be normalized once, using 784 features for RF/MLP and 28 by 28 by 1 for CNN.
