$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$linuxRoot = (& wsl -d Ubuntu-24.04 -u root -- wslpath -a $root).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Ubuntu-24.04 is not available. See docs/local_setup.md.' }
Write-Host 'Open http://localhost:8501 in your browser. Press Ctrl+C to stop.'
& wsl -d Ubuntu-24.04 -u root --cd $linuxRoot -- env CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=2 TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=2 /opt/mnist-app/bin/python -m streamlit run "$linuxRoot/app.py" --server.address 0.0.0.0 --server.port 8501 --server.headless true
