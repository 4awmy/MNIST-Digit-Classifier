param([ValidateSet('all','rf','mlp','cnn')][string]$Model = 'all')
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$linuxRoot = (& wsl -d Ubuntu-24.04 -u root -- wslpath -a $root).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Ubuntu-24.04 is not available. See docs/local_setup.md.' }
& wsl -d Ubuntu-24.04 -u root -- env KERAS_HOME=/opt/mnist-cache/keras MPLBACKEND=Agg TF_CPP_MIN_LOG_LEVEL=2 /opt/mnist-env/bin/python "$linuxRoot/tools/run_training.py" --model $Model
if ($LASTEXITCODE -ne 0) { throw 'Training failed. Inspect results/<model>/training.log.' }
