"""Internal parameter validation, model I/O, format conversion, and inspection.

models.factory constructs models and infers structures. Avoid importing model
base classes here to prevent a base -> utils -> base dependency cycle.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Union, Optional, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from .base import WhiteBoxDNN


def prepare_parameters(weights, biases, shapes, *, layout='out_in', device=None, dtype=None):
    """Validate every layer and return independent Tensors without mutating inputs."""
    if layout not in ('out_in', 'in_out'):
        raise ValueError(f'Unknown weight layout: {layout}')
    weights, biases, shapes = list(weights), list(biases), list(shapes)
    if len(weights) != len(shapes) or len(biases) != len(shapes):
        raise ValueError('Weight/bias counts must match the number of layers')
    prepared = []
    for index, (weight, bias, (input_dim, output_dim)) in enumerate(zip(weights, biases, shapes)):
        weight = torch.as_tensor(weight).detach()
        bias = torch.as_tensor(bias).detach()
        if weight.ndim != 2:
            raise ValueError(f'Layer {index}: weight must be a matrix')
        if layout == 'in_out':
            weight = weight.t()
        if tuple(weight.shape) != (output_dim, input_dim) or tuple(bias.shape) != (output_dim,):
            raise ValueError(f'Layer {index}: expected weight {(output_dim, input_dim)} and bias {(output_dim,)}')
        weight = weight.to(device=device, dtype=dtype).clone().contiguous()
        bias = bias.to(device=device, dtype=dtype).clone()
        if not torch.isfinite(weight).all() or not torch.isfinite(bias).all():
            raise ValueError(f'Layer {index}: parameters must be finite')
        prepared.append((weight, bias))
    return prepared


def set_parameters(model, weights, biases, *, layout='out_in'):
    """Update parameters in place while preserving Parameter identity, device, and dtype.

    If any layer fails validation, no layer is written.
    """
    layers = list(model.fcs)
    pairs = prepare_parameters(
        weights, biases, [(layer.in_features, layer.out_features) for layer in layers], layout=layout,
    )
    converted = []
    for layer, (weight, bias) in zip(layers, pairs):
        weight = weight.to(layer.weight).clone()
        bias = bias.to(layer.bias).clone()
        if not torch.isfinite(weight).all() or not torch.isfinite(bias).all():
            raise ValueError('Parameters overflow the model dtype')
        converted.append((weight, bias))
    with torch.no_grad():
        for layer, (weight, bias) in zip(layers, converted):
            layer.weight.copy_(weight)
            layer.bias.copy_(bias)


def get_parameters(model, *, layout='out_in', layers=None):
    """Export independent NumPy copies; modifying them does not modify the model."""
    if layout not in ('out_in', 'in_out'):
        raise ValueError(f'Unknown weight layout: {layout}')
    indices = range(len(model.fcs)) if layers is None else layers
    weights, biases = [], []
    for index in indices:
        layer = model.fcs[index]
        weight = layer.weight.detach()
        if layout == 'in_out':
            weight = weight.t()
        weights.append(weight.cpu().numpy().copy())
        biases.append(layer.bias.detach().cpu().numpy().copy())
    return weights, biases


def load_model_weights(
    model: nn.Module,
    filepath: Union[str, Path],
    map_location: str = 'cpu',
    strict: bool = True
) -> nn.Module:
    """Load checkpoint weights into a model with the requested map_location and strictness."""
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")

    state_dict = torch.load(filepath, map_location=map_location, weights_only=True)
    model.load_state_dict(state_dict, strict=strict)

    return model


def save_basemodel_compatible(
    model: WhiteBoxDNN,
    filepath: Union[str, Path],
    include_structure: bool = True
) -> None:
    """Save a WhiteBoxDNN-compatible checkpoint, optionally including structure metadata."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    state_dict = {}

    for i, layer in enumerate(model.fcs):
        if isinstance(layer, nn.Linear):
            state_dict[f'fcs.{i}.weight'] = layer.weight.data
            state_dict[f'fcs.{i}.bias'] = layer.bias.data

    if include_structure:
        state_dict['structure'] = torch.tensor(model.structure)

    torch.save(state_dict, filepath)


def load_basemodel_compatible(
    model: WhiteBoxDNN,
    filepath: Union[str, Path],
    map_location: str = 'cpu'
) -> WhiteBoxDNN:
    """Load weights from a WhiteBoxDNN-compatible checkpoint into a model."""
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")

    state_dict = torch.load(filepath, map_location=map_location, weights_only=True)

    weights = []
    biases = []

    for i in range(len(model.fcs)):
        weight_key = f'fcs.{i}.weight'
        bias_key = f'fcs.{i}.bias'

        if weight_key in state_dict and bias_key in state_dict:
            weights.append(state_dict[weight_key])
            biases.append(state_dict[bias_key])

    model.set_weights_bias(weights, biases)

    return model


def get_model_info(model: WhiteBoxDNN) -> Dict:
    """Return a dictionary describing a WhiteBoxDNN model."""
    info = {
        'structure': model.structure,
        'num_layers': len(model.fcs),
        'total_params': sum(p.numel() for p in model.parameters()),
        'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'device': None,
        'dtype': None
    }

    reference = next(model.parameters(), None)
    if reference is None:
        reference = next(model.buffers(), None)
    if reference is not None:
        info['device'] = reference.device
        info['dtype'] = reference.dtype

    layer_info = []
    for i, layer in enumerate(model.fcs):
        if isinstance(layer, nn.Linear):
            layer_info.append({
                'layer_id': i,
                'in_features': layer.in_features,
                'out_features': layer.out_features,
                'has_bias': layer.bias is not None
            })

    info['layers'] = layer_info

    return info


def print_model_summary(model: WhiteBoxDNN) -> None:
    """Print a summary of a WhiteBoxDNN model."""
    info = get_model_info(model)

    print("=" * 60)
    print("Model summary")
    print("=" * 60)
    print(f"Network structure: {info['structure']}")
    print(f"Layer count: {info['num_layers']}")
    print(f"Total parameters: {info['total_params']:,}")
    print(f"Trainable parameters: {info['trainable_params']:,}")
    print(f"Device: {info['device']}")
    print(f"Data type: {info['dtype']}")
    print("-" * 60)
    print("Layer details:")
    for layer in info['layers']:
        print(f"  Layer {layer['layer_id']}: "
              f"{layer['in_features']} -> {layer['out_features']}"
              f"{' (with bias)' if layer['has_bias'] else ''}")
    print("=" * 60)


def transform_checkpoint(old_path, new_path, structure=None):
    """Convert fc{i}.weight checkpoint keys to fcs.{i}.weight keys.

    structure specifies the expected layer widths.
    """
    if structure is None:
        structure = [64, 64, 64, 64, 64, 10]

    checkpoint = torch.load(old_path)

    new_state_dict = {}
    from .base import WhiteBoxDNN
    model = WhiteBoxDNN(structure)

    for i in range(len(structure) - 1):
        old_weight_key = f'fc{i + 1}.weight'
        old_bias_key = f'fc{i + 1}.bias'
        new_weight_key = f'fcs.{i}.weight'
        new_bias_key = f'fcs.{i}.bias'

        if old_weight_key in checkpoint:
            new_state_dict[new_weight_key] = checkpoint[old_weight_key]
        if old_bias_key in checkpoint:
            new_state_dict[new_bias_key] = checkpoint[old_bias_key]

    torch.save(new_state_dict, new_path)
    model.load_state_dict(new_state_dict)
    print(f"Checkpoint transformed and saved to {new_path}")
    return new_state_dict


def clean_structure_from_pth(input_path, output_path=None, backup=True):
    """Remove structure metadata from a .pth checkpoint.

    output_path=None overwrites the input; backup optionally preserves a copy.
    Returns whether cleanup succeeded.
    """
    import shutil

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    if output_path is None:
        output_path = input_path
        if backup:
            backup_path = input_path.with_suffix('.pth.bak')
            print(f"Backing up input to: {backup_path}")
            shutil.copy2(input_path, backup_path)
    else:
        output_path = Path(output_path)

    print(f"Loading: {input_path}")
    state_dict = torch.load(input_path, map_location='cpu', weights_only=False)

    if 'structure' not in state_dict:
        print("No structure field found; cleanup is unnecessary")
        return False

    structure = state_dict['structure']
    print(f"Detected structure field: {structure}")
    del state_dict['structure']
    print("Removed structure field")

    print(f"\nRetained fields ({len(state_dict)}):")
    for key in sorted(state_dict.keys()):
        if isinstance(state_dict[key], torch.Tensor):
            print(f"  - {key}: shape={state_dict[key].shape}")
        else:
            print(f"  - {key}: {type(state_dict[key])}")

    print(f"\nSaving to: {output_path}")
    torch.save(state_dict, output_path)
    print("Cleanup complete")
    return True


def batch_clean_structure(directory, pattern="*.pth", backup=True):
    """Remove structure metadata from checkpoint files matching a directory pattern."""
    directory = Path(directory)
    files = list(directory.glob(pattern))

    if not files:
        print(f"In {directory}, no files match {pattern}")
        return

    print(f"Found {len(files)} files to process\n")

    cleaned_count = 0
    for file_path in files:
        print(f"\n{'='*60}")
        print(f"Processing file: {file_path.name}")
        print('='*60)
        try:
            if clean_structure_from_pth(file_path, backup=backup):
                cleaned_count += 1
        except Exception as e:
            print(f"Processing failed: {e}")

    print(f"\n{'='*60}")
    print(f"Processing complete; cleaned {cleaned_count}/{len(files)} files")
    print('='*60)


FORMAT_PTH = 'pth'
FORMAT_KERAS = 'keras'

FORMAT_EXTENSIONS = {
    FORMAT_PTH: '.pth',
    FORMAT_KERAS: '.keras',
}

# Support fc1.weight and fcs.0.weight checkpoint key formats.
_OLD_KEY_PREFIX = 'fc'
_NEW_KEY_PREFIX = 'fcs.'


def detect_format(filepath: Union[str, Path]) -> str:
    """Detect pth or keras format from the extension; reject unsupported formats."""
    ext = Path(filepath).suffix.lower()
    if ext == '.pth':
        return FORMAT_PTH
    elif ext in ('.keras', '.h5', '.hdf5'):
        return FORMAT_KERAS
    else:
        raise ValueError(f"Unsupported model format: {ext}; supported: {list(FORMAT_EXTENSIONS.values())}")


def detect_checkpoint_format(state_dict: Dict) -> str:
    """Return new for fcs.0.weight keys or old for fc1.weight keys."""
    keys = list(state_dict.keys())
    if any(k.startswith(_NEW_KEY_PREFIX) for k in keys):
        return 'new'
    elif any(k.startswith(_OLD_KEY_PREFIX) and k[2:3].isdigit() for k in keys):
        return 'old'
    else:
        return 'unknown'


def infer_dtype_from_pth(filepath: Union[str, Path]) -> str:
    """Infer f32 or f64 precision from a PyTorch checkpoint."""
    state_dict = torch.load(filepath, map_location='cpu', weights_only=False)
    for v in state_dict.values():
        if isinstance(v, torch.Tensor):
            return 'f64' if v.dtype == torch.float64 else 'f32'
    return 'f32'


def structure_to_str(structure: List[int]) -> str:
    """Format layer widths, compressing repeated widths with x counts.

    For example, [64, 64, 64, 10] becomes 64x3_10.
    """
    if not structure:
        return ''

    parts = []
    i = 0
    while i < len(structure):
        val = structure[i]
        count = 1
        while i + count < len(structure) and structure[i + count] == val:
            count += 1
        if count > 1:
            parts.append(f'{val}x{count}')
        else:
            parts.append(str(val))
        i += count

    return '_'.join(parts)


def structure_from_str(struct_str: str) -> List[int]:
    """Parse a structure string with optional x counts into layer widths.

    For example, 64x3_10 becomes [64, 64, 64, 10].
    """
    structure = []
    for part in struct_str.split('_'):
        if 'x' in part:
            val, count = part.split('x')
            structure.extend([int(val)] * int(count))
        else:
            structure.append(int(part))
    return structure


def generate_weight_filename(structure: List[int], fmt: str = FORMAT_PTH,
                             suffix: str = '') -> str:
    """Build a standardized weight filename from structure, format, and optional suffix.

    For example, pth_64x5_10.pth.
    """
    ext = FORMAT_EXTENSIONS.get(fmt, '.pth')
    struct_str = structure_to_str(structure)
    return f"{fmt}_{struct_str}{suffix}{ext}"


def generate_keras_filename(dataset: str, structure: List[int],
                            precision: str = 'f64') -> str:
    """Build a Keras filename from dataset, structure, and precision.

    For example, keras_cifar10_768_64x5_10_f64.keras.
    """
    struct_str = structure_to_str(structure)
    return f"keras_{dataset}_{struct_str}_{precision}.keras"


def parse_weight_filename(filename: str) -> Dict:
    """Parse a standardized filename into format, structure, and suffix fields."""
    stem = Path(filename).stem
    parts = stem.split('_')

    result = {'format': parts[0], 'structure': [], 'suffix': ''}

    if parts[0] == FORMAT_PTH:
        # A numeric final component belongs to the structure; other components are suffixes.
        struct_parts = []
        suffix_parts = []
        for p in parts[1:]:
            if suffix_parts or (not any(c.isdigit() for c in p) and 'x' not in p):
                suffix_parts.append(p)
            else:
                struct_parts.append(p)
        result['structure'] = structure_from_str('_'.join(struct_parts))
        result['suffix'] = '_'.join(suffix_parts)
    elif parts[0] == FORMAT_KERAS:
        # keras_cifar10_768_64x5_10_f64
        result['dataset'] = parts[1]
        # The final component encodes precision; intervening components encode structure.
        result['precision'] = parts[-1]
        result['structure'] = structure_from_str('_'.join(parts[2:-1]))

    return result


def load_pth(filepath: Union[str, Path],
            map_location: str = 'cpu',
            weights_only: bool = False) -> Dict:
    """Load a PyTorch state_dict with optional device mapping and weights-only loading."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")
    return torch.load(filepath, map_location=map_location, weights_only=weights_only)


def load_keras(filepath: Union[str, Path]):
    """Load a Keras model from a .keras or .h5 file."""
    import tensorflow as tf
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")
    return tf.keras.models.load_model(str(filepath))


def load_model(filepath: Union[str, Path], **kwargs):
    """Select a loader by file format and forward keyword options to it."""
    fmt = detect_format(filepath)
    if fmt == FORMAT_PTH:
        return load_pth(filepath, **kwargs)
    elif fmt == FORMAT_KERAS:
        return load_keras(filepath, **kwargs)
    else:
        raise ValueError(f"Unsupported format: {fmt}")


def save_pth(model_or_state_dict, filepath: Union[str, Path],
             state_dict_only: bool = True) -> None:
    """Save a PyTorch model or state_dict, optionally exporting only state_dict."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(model_or_state_dict, dict):
        torch.save(model_or_state_dict, filepath)
    elif isinstance(model_or_state_dict, torch.nn.Module):
        if state_dict_only:
            torch.save(model_or_state_dict.state_dict(), filepath)
        else:
            torch.save(model_or_state_dict, filepath)
    else:
        raise TypeError(f"Unsupported type: {type(model_or_state_dict)}")


def save_keras(model, filepath: Union[str, Path]) -> None:
    """Save a model in Keras format."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(filepath))


def save_model(model, filepath: Union[str, Path], **kwargs) -> None:
    """Select a saver by file format and forward keyword options to it."""
    fmt = detect_format(filepath)
    if fmt == FORMAT_PTH:
        save_pth(model, filepath, **kwargs)
    elif fmt == FORMAT_KERAS:
        save_keras(model, filepath, **kwargs)
    else:
        raise ValueError(f"Unsupported format: {fmt}")


def save_basemodel(model: WhiteBoxDNN, filepath: Union[str, Path],
                   include_structure: bool = False) -> None:
    """Save a WhiteBoxDNN checkpoint, optionally including a structure Tensor."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    state_dict = {}
    for i, layer in enumerate(model.fcs):
        if isinstance(layer, torch.nn.Linear):
            state_dict[f'fcs.{i}.weight'] = layer.weight.data
            state_dict[f'fcs.{i}.bias'] = layer.bias.data

    if include_structure:
        state_dict['structure'] = torch.tensor(model.structure)

    torch.save(state_dict, filepath)


def load_basemodel(model: WhiteBoxDNN, filepath: Union[str, Path],
                   map_location: str = 'cpu') -> WhiteBoxDNN:
    """Load a WhiteBoxDNN checkpoint, automatically converting supported key formats."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")

    state_dict = torch.load(filepath, map_location=map_location, weights_only=True)
    fmt = detect_checkpoint_format(state_dict)

    if fmt == 'new':
        # Load fcs.0.weight keys directly.
        weights, biases = [], []
        for i in range(len(model.fcs)):
            w_key = f'fcs.{i}.weight'
            b_key = f'fcs.{i}.bias'
            if w_key in state_dict and b_key in state_dict:
                weights.append(state_dict[w_key])
                biases.append(state_dict[b_key])
        model.set_weights_bias(weights, biases)
    elif fmt == 'old':
        # Convert fc1.weight keys before loading.
        weights, biases = [], []
        for i in range(len(model.fcs)):
            w_key = f'fc{i + 1}.weight'
            b_key = f'fc{i + 1}.bias'
            if w_key in state_dict and b_key in state_dict:
                weights.append(state_dict[w_key])
                biases.append(state_dict[b_key])
        model.set_weights_bias(weights, biases)
    else:
        raise ValueError(f"Unrecognized checkpoint format; check parameter key names")

    return model


def convert_old_to_new_format(old_path: Union[str, Path],
                              new_path: Union[str, Path] = None,
                              structure: List[int] = None) -> Dict:
    """Convert fc{i}.weight keys to fcs.{i}.weight keys and return the state_dict.

    new_path=None skips saving. An optional structure validates layer widths.
    """
    state_dict = torch.load(old_path, map_location='cpu', weights_only=False)

    if structure is None:
        from .factory import infer_structure_from_state_dict
        structure = infer_structure_from_state_dict(state_dict)

    new_state_dict = {}
    for i in range(len(structure) - 1):
        old_w = f'fc{i + 1}.weight'
        old_b = f'fc{i + 1}.bias'
        new_w = f'fcs.{i}.weight'
        new_b = f'fcs.{i}.bias'

        if old_w in state_dict:
            new_state_dict[new_w] = state_dict[old_w]
        if old_b in state_dict:
            new_state_dict[new_b] = state_dict[old_b]

    if new_path is not None:
        save_pth(new_state_dict, new_path)
        print(f"Converted: {old_path} -> {new_path}")

    return new_state_dict


# Store each model under its own assets/model_name directory.
#   assets/
#   ├── keras_cifar10_768_64x5_10_f64/
#   │   ├── model.keras
#   │   ├── x_test.npy
#   │   └── dual_points/
#   ├── pth_64x5_10/
#   │   └── model.pth
#   └── ...

def get_assets_dir() -> Path:
    """Return the assets directory path."""
    return Path(__file__).parent.parent / 'assets'


def get_model_dir(model_name: str) -> Path:
    """Return assets/model_name for the requested model."""
    return get_assets_dir() / model_name


def list_models() -> List[str]:
    """List model directory names under assets."""
    assets_dir = get_assets_dir()
    if not assets_dir.exists():
        return []
    return sorted(
        d.name for d in assets_dir.iterdir()
        if d.is_dir() and not d.name.startswith('.')
    )


def find_model_file(model_name: str, filename: str = 'model.pth') -> Optional[Path]:
    """Find a file in a model directory, or return None if absent."""
    filepath = get_model_dir(model_name) / filename
    return filepath if filepath.exists() else None


def find_weight(structure: List[int], fmt: str = FORMAT_PTH,
                suffix: str = '') -> Optional[Path]:
    """Find a weight file matching structure, format, and directory suffix, or return None."""
    dir_name = generate_weight_filename(structure, fmt, suffix)
    dir_name = Path(dir_name).stem
    filepath = get_model_dir(dir_name) / 'model.pth'
    return filepath if filepath.exists() else None


def list_weights(fmt: Optional[str] = None) -> List[Path]:
    """List model weight files under assets, optionally filtering by format."""
    assets_dir = get_assets_dir()
    if not assets_dir.exists():
        return []

    files = []
    for model_dir in assets_dir.iterdir():
        if not model_dir.is_dir() or model_dir.name.startswith('.'):
            continue
        if fmt:
            ext = FORMAT_EXTENSIONS.get(fmt, '.pth')
            files.extend(model_dir.glob(f'*{ext}'))
        else:
            for e in FORMAT_EXTENSIONS.values():
                files.extend(model_dir.glob(f'*{e}'))
    return sorted(files)


def inspect_weight(filepath: Union[str, Path]) -> Dict:
    """Inspect a weight file and return its format, structure, dtype, and keys."""
    filepath = Path(filepath)
    fmt = detect_format(filepath)

    info = {'path': str(filepath), 'filename': filepath.name, 'format': fmt}

    if fmt == FORMAT_PTH:
        state_dict = torch.load(filepath, map_location='cpu', weights_only=False)
        info['checkpoint_format'] = detect_checkpoint_format(state_dict)
        from .factory import infer_structure_from_state_dict
        info['structure'] = infer_structure_from_state_dict(state_dict)
        info['num_keys'] = len(state_dict)

        for v in state_dict.values():
            if isinstance(v, torch.Tensor):
                info['dtype'] = str(v.dtype)
                break

    return info
