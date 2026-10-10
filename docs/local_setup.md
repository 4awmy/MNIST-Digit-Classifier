# Local setup

## This laptop

Ubuntu 24.04 runs under WSL2. Its training environment is `/opt/mnist-env`, its kernel is `mnist-local`, and the MNIST download cache is `/opt/mnist-cache/keras`. A separate CPU-only app environment at `/opt/mnist-app` matches the Cloud dependencies. The WSL disk lives under the Codex working directory; it is separate from the Git repository. NVIDIA's Windows driver supplies GPU access to WSL.

From the Desktop Assignment1 folder in PowerShell:

```powershell
.\tools\train_local.ps1             # Train all three sequentially
.\tools\train_local.ps1 -Model mlp  # Or rerun one family
.\tools\run_app.ps1                 # Open http://localhost:8501
```

The runner uses CPU for Random Forest and requires a visible GPU for MLP/CNN. It preserves executed notebooks, logs and run status. A failed notebook keeps its diagnostic outputs and does not replace the source notebook with a partially successful run. Rerunning a family replaces its previous results; make a backup if you want to compare runs.

## Another computer

The saved models can run on CPU; a GPU is only needed for the chosen local training workflow. Use Python 3.12, create a virtual environment, and install the minimal application dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

For Linux/WSL GPU training, install `requirements-gpu.txt` in a separate virtual environment, register an `mnist-local` ipykernel with `python -m ipykernel install --sys-prefix --name mnist-local`, and run `python tools/run_training.py`. `requirements-training-lock.txt` records this laptop's complete working environment, including Linux CUDA packages; it is not the Streamlit deployment dependency file. For CPU notebook training, use `requirements-training.txt` and open each notebook in a fresh Jupyter kernel rather than the GPU-enforcing runner.

The Linux runner adds NVIDIA wheel library directories to `LD_LIBRARY_PATH` before starting its worker and notebook kernel. On this laptop, this was needed for TensorFlow to detect the installed CUDA libraries. `python tools/setup_gpu_links.py` also configures the virtual-environment library/`ptxas` links described in the official TensorFlow guide. These changes affect the training environment; the separate CPU app environment does not use them. For manual GPU notebook execution outside the runner, configure the same library search path in the shell that launches Jupyter.

Modern TensorFlow GPU support on Windows uses WSL2. Follow the official [TensorFlow pip installation guide](https://www.tensorflow.org/install/pip) and [Microsoft WSL installation guide](https://learn.microsoft.com/en-us/windows/wsl/install) when preparing another machine. Verify `tf.config.list_physical_devices('GPU')` and an actual GPU operation before starting a long run. Do not install a separate Linux NVIDIA display driver inside WSL.
