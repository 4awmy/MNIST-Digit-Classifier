"""Run each notebook in a fresh local kernel and preserve outputs even on failure."""
import argparse
import json
import os
import subprocess
import sys
import sysconfig
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = {'rf': 'Random_forest_.ipynb', 'mlp': 'Mlp.ipynb', 'cnn': 'CNN.ipynb'}


def run_one(name, kernel):
    import nbformat
    from nbclient import NotebookClient

    source = ROOT / NAMES[name]
    if not source.exists():
        source = ROOT / 'notebooks' / NAMES[name]
    notebook = nbformat.read(source, as_version=4)
    result_dir = ROOT / 'results' / name
    result_dir.mkdir(parents=True, exist_ok=True)
    executed = result_dir / 'executed.ipynb'
    status = {'model': name, 'status': 'running', 'started_unix': time.time()}
    status_path = result_dir / 'run_status.json'
    status_path.write_text(json.dumps(status, indent=2), encoding='utf-8')

    def progress(cell, cell_index, **kwargs):
        if cell.cell_type == 'code':
            heading = cell.source.strip().splitlines()[0] if cell.source.strip() else '(empty)'
            print(f'[{name}] cell {cell_index+1}: {heading}', flush=True)

    class LoggedNotebookClient(NotebookClient):
        def process_message(self, msg, cell, cell_index):
            if msg['msg_type'] == 'stream':
                output = msg['content']['text']
                # Download progress uses backspaces; retain it in notebook outputs,
                # while keeping the plain-text terminal log readable.
                if '\b' not in output:
                    print(output, end='', flush=True)
            return super().process_message(msg, cell, cell_index)

    client = LoggedNotebookClient(notebook, kernel_name=kernel, timeout=1800,
                            resources={'metadata': {'path': str(ROOT)}},
                            on_cell_start=progress, record_timing=True)
    try:
        client.execute()
    except BaseException as exc:
        status.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    else:
        status['status'] = 'complete'
    finally:
        status['finished_unix'] = time.time()
        status_path.write_text(json.dumps(status, indent=2), encoding='utf-8')
        nbformat.write(notebook, executed)
        if status['status'] == 'complete':
            nbformat.write(notebook, source)
        print(f'[{name}] {status["status"]}; executed notebook: {executed}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=[*NAMES, 'all'], default='all')
    parser.add_argument('--kernel', default='mnist-local')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        run_one(args.model, args.kernel)
        return
    selected = list(NAMES) if args.model == 'all' else [args.model]
    for name in selected:
        environment = os.environ.copy()
        if sys.platform.startswith('linux'):
            # CUDA wheels keep libraries outside the system loader's search path.
            nvidia = Path(sysconfig.get_paths()['purelib']) / 'nvidia'
            libraries = [str(path) for path in sorted(nvidia.glob('*/lib')) if path.is_dir()]
            if libraries:
                environment['LD_LIBRARY_PATH'] = ':'.join(libraries + [environment.get('LD_LIBRARY_PATH', '')])
        environment['PATH'] = str(Path(sys.executable).parent) + os.pathsep + environment.get('PATH', '')
        if name == 'rf':
            environment['CUDA_VISIBLE_DEVICES'] = '-1'
        else:
            environment['MNIST_REQUIRE_GPU'] = '1'
        # Leave CPU capacity for Windows and bound TensorFlow's input workers.
        environment.setdefault('TF_NUM_INTRAOP_THREADS', '8')
        environment.setdefault('TF_NUM_INTEROP_THREADS', '2')
        environment.setdefault('OMP_NUM_THREADS', '8')
        environment['PYTHONUNBUFFERED'] = '1'
        command = [sys.executable, str(Path(__file__).resolve()), '--model', name, '--kernel', args.kernel, '--worker']
        log = ROOT / 'results' / name / 'training.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('w', encoding='utf-8') as handle:
            process = subprocess.Popen(command, env=environment, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                print(line, end='', flush=True)
                handle.write(line)
                handle.flush()
            if process.wait():
                raise SystemExit(f'{name} failed. Inspect {log}; earlier completed models were preserved.')
    if args.model == 'all':
        subprocess.run([sys.executable, str(ROOT / 'tools' / 'compare_results.py')], check=True)


if __name__ == '__main__':
    main()
