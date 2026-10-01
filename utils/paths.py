"""Resolve project paths independently of the current working directory."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = PROJECT_ROOT / 'runtime'
RESULTS_ROOT = PROJECT_ROOT / 'results'


def results_path(path=''):
    """Resolve experiment outputs under results; absolute destinations stay explicit."""
    path = Path(path).expanduser()
    if path.is_absolute():
        return path.resolve()
    parts = path.parts
    if parts and parts[0].lower() == 'results':
        parts = parts[1:]
    result = RESULTS_ROOT.joinpath(*parts).resolve()
    if not result.is_relative_to(RESULTS_ROOT.resolve()):
        raise ValueError('Relative output path must stay inside the project results directory')
    return result


def project_path(path):
    """Resolve relative asset paths from the project root; keep absolute paths unchanged."""
    path = Path(path).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def runtime_path(path=''):
    """Resolve relative outputs under runtime, accepting a runtime/ prefix or absolute path."""
    path = Path(path).expanduser()
    if path.is_absolute():
        return path.resolve()
    parts = path.parts
    if parts and parts[0].lower() == 'runtime':
        parts = parts[1:]
    result = RUNTIME_ROOT.joinpath(*parts).resolve()
    if not result.is_relative_to(RUNTIME_ROOT.resolve()):
        raise ValueError('Relative output path must stay inside the project runtime directory')
    return result
