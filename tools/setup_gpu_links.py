"""Apply TensorFlow's documented pip CUDA library links inside the active venv."""
import importlib.util
from pathlib import Path
import sys

if sys.prefix == sys.base_prefix:
    raise SystemExit('Run this helper inside the GPU training virtual environment.')
spec = importlib.util.find_spec('tensorflow')
if spec is None or spec.origin is None:
    raise SystemExit('Install the GPU training dependencies in a virtual environment first.')
tensorflow = Path(spec.origin).parent
nvidia = tensorflow.parent / 'nvidia'
libraries = sorted(nvidia.glob('*/lib/*.so*'))
if not libraries:
    raise SystemExit('NVIDIA pip libraries are missing; this helper is for the GPU training environment only.')
links = [(library, tensorflow / library.name) for library in libraries]
ptxas = nvidia / 'cuda_nvcc' / 'bin' / 'ptxas'
if ptxas.is_file():
    links.append((ptxas, Path(sys.prefix) / 'bin' / 'ptxas'))
for source, target in links:
    if target.is_symlink():
        target.unlink()
    elif target.exists():
        raise SystemExit(f'Refusing to replace an existing regular file: {target}')
    target.symlink_to(source)
print(f'Configured {len(links)} CUDA library/compiler links inside {sys.prefix}.')
