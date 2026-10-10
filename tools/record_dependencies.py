"""Pin direct requirements to the versions actually used for training and app checks."""
from importlib.metadata import version
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
app = ['streamlit','keras','scikit-learn','joblib','numpy','pandas','pillow','h5py']
training = ['tensorflow','keras','scikit-learn','joblib','numpy','pandas','matplotlib','pillow','nbclient','nbformat','ipykernel']
versions = {name:version(name) for name in set(app+training)}
def pins(names): return '\n'.join(f'{name}=={versions[name]}' for name in names) + '\n'
(ROOT/'requirements.txt').write_text('# CPU application dependencies; tested with Python 3.12.\n' +
    f'tensorflow-cpu=={versions["tensorflow"]}\n' + pins(app), encoding='utf-8')
(ROOT/'requirements-training.txt').write_text('# Direct notebook/runner requirements from the completed local environment.\n'+pins(training), encoding='utf-8')
(ROOT/'requirements-gpu.txt').write_text('# Local WSL GPU training; use a separate environment from the CPU app.\n-c requirements-cuda.txt\n-r requirements-training.txt\n'+f'tensorflow[and-cuda]=={versions["tensorflow"]}\n', encoding='utf-8')
(ROOT/'results'/'dependency_versions.json').write_text(json.dumps(versions, indent=2)+'\n', encoding='utf-8')
print('Recorded tested direct dependency versions.')
