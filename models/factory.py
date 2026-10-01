"""Construct, initialize, and assemble models for external callers.

models.utils handles internal parameter operations and file formats.
"""
import torch
from pathlib import Path
from typing import Union, Dict, List

from .base import BlackBoxDNN, WhiteBoxDNN, RecoveryModel


def infer_structure_from_pth(filepath: Union[str, Path]) -> List[int]:
    """Infer a network structure list from a PyTorch checkpoint."""
    state_dict = torch.load(filepath, map_location='cpu', weights_only=True)
    return infer_structure_from_state_dict(state_dict)


def infer_structure_from_state_dict(state_dict: Dict) -> List[int]:
    """Infer a network structure list from a state_dict."""
    from .utils import detect_checkpoint_format
    _NEW_KEY_PREFIX, _OLD_KEY_PREFIX = 'fcs.', 'fc'
    fmt = detect_checkpoint_format(state_dict)

    structure = []
    if fmt == 'new':
        i = 0
        while f'{_NEW_KEY_PREFIX}{i}.weight' in state_dict:
            w = state_dict[f'{_NEW_KEY_PREFIX}{i}.weight']
            if i == 0:
                structure.append(w.shape[1])  # in_features of first layer
            structure.append(w.shape[0])  # out_features
            i += 1
    elif fmt == 'old':
        i = 1
        while f'{_OLD_KEY_PREFIX}{i}.weight' in state_dict:
            w = state_dict[f'{_OLD_KEY_PREFIX}{i}.weight']
            if i == 1:
                structure.append(w.shape[1])
            structure.append(w.shape[0])
            i += 1
    else:
        raise ValueError(f"Unrecognized checkpoint key format")

    return structure


def initialize_cheat_model(
    structure: list,
    weights_path: Union[str, Path] = './assets/pth_64x5_10/model.pth',
    use_cuda: bool = True,
    double_precision: bool = True
) -> WhiteBoxDNN:
    """Initialize and load a white-box model for validation.

    structure specifies layer widths; weights_path supplies the checkpoint.
    use_cuda and double_precision select device and precision.
    """
    return create_and_load_model(structure, weights_path, use_cuda=use_cuda,
                                 double_precision=double_precision)


def initialize_recovery_model(
    input_dim: int,
    weights_path: Union[str, Path] = './assets/pth_64x5_10/model.pth',
    use_cuda: bool = True,
    double_precision: bool = True
) -> RecoveryModel:
    """Initialize an empty recovered prefix without reading target weights.

    weights_path is accepted for compatibility but ignored. input_dim sets the
    identity prefix width; use_cuda and double_precision select placement.
    """
    recovery_model = RecoveryModel(input_dim, device='cpu', with_relu=True)

    if double_precision:
        recovery_model = recovery_model.double()
    if use_cuda and torch.cuda.is_available():
        recovery_model = recovery_model.cuda()

    return recovery_model


def initialize_models(
    structure: list,
    weights_path: Union[str, Path] = './assets/pth_64x5_10/model.pth',
    use_cuda: bool = True,
    double_precision: bool = True
):
    """Initialize a WhiteBoxDNN and a RecoveryModel.

    Returns (cheat_model, recovery_model).
    """
    cheat_model = initialize_cheat_model(
        structure, weights_path, use_cuda, double_precision
    )
    recovery_model = initialize_recovery_model(
        cheat_model.input_dim, weights_path, use_cuda, double_precision
    )
    return cheat_model, recovery_model


def create_and_load_model(
    structure: list = None,
    weights_path: Union[str, Path] = None,
    model_class: type = WhiteBoxDNN,
    map_location: str = 'cpu',
    use_cuda: bool = True,
    double_precision: bool = False,
    *,
    device=None,
    dtype=None,
    strict: bool = True,
) -> WhiteBoxDNN:
    """Read a checkpoint once, determine structure, and create the loaded model.

    structure=None infers widths from weights. Explicit device and dtype override
    use_cuda and double_precision. strict controls checkpoint key validation.
    """
    if weights_path is None:
        raise ValueError('weights_path is required')
    from .utils import detect_checkpoint_format

    state_dict = torch.load(weights_path, map_location=map_location, weights_only=True)
    if structure is None:
        structure = infer_structure_from_state_dict(state_dict)
        if not structure:
            raise ValueError(f'Cannot infer model structure from {weights_path}')
    model = model_class(structure, device='cpu')
    if dtype is not None or double_precision:
        model = model.to(dtype=dtype if dtype is not None else torch.float64)
    checkpoint = dict(state_dict)
    checkpoint.pop('structure', None)
    if detect_checkpoint_format(checkpoint) == 'old':
        for index in range(len(structure) - 1):
            for parameter in ('weight', 'bias'):
                old_key = f'fc{index + 1}.{parameter}'
                if old_key in checkpoint:
                    checkpoint[f'fcs.{index}.{parameter}'] = checkpoint.pop(old_key)
    model.load_state_dict(checkpoint, strict=strict)
    if device is not None:
        model = model.to(device=device)
    elif use_cuda and torch.cuda.is_available():
        model = model.cuda()
    return model


def create_model(path, structure=None, device='cuda'):
    """Load a WhiteBoxDNN and return (model, cheat_solution)."""
    cheat_model = create_and_load_model(structure, path)

    cheat_solution = [x.cpu().detach().numpy() for x in cheat_model.parameters()][::2]

    cheat_model.double()
    cheat_model.to(device=device)

    return cheat_model, cheat_solution


def build_partial_recovery_model(
    full_structure: list,
    weights: list,
    biases: list,
    n_layers: int,
    use_cuda: bool = True,
    double_precision: bool = True,
    *,
    layout: str = 'in_out'
) -> RecoveryModel:
    """Build a RecoveryModel from the first n_layers of known parameters.

    Weights use transposed NumPy layout. n_layers=0 creates an identity prefix.
    use_cuda and double_precision select placement.
    """
    from numbers import Integral
    from .utils import prepare_parameters

    if len(full_structure) < 2 or any(isinstance(n, bool) or not isinstance(n, Integral) or n <= 0 for n in full_structure):
        raise ValueError('full_structure must describe a complete network')
    if isinstance(n_layers, bool) or not isinstance(n_layers, Integral) or not 0 <= n_layers <= len(full_structure) - 2:
        raise ValueError('n_layers must select hidden layers only')
    input_dim = full_structure[0]
    dtype = torch.float64 if double_precision else torch.get_default_dtype()
    device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'
    recovery_model = RecoveryModel(input_dim, device=device, dtype=dtype, with_relu=True)
    pairs = prepare_parameters(
        weights[:n_layers], biases[:n_layers],
        list(zip(full_structure, full_structure[1:]))[:n_layers],
        layout=layout, device=device, dtype=dtype,
    )
    for weight, bias in pairs:
        recovery_model.append_layer(weight, bias)

    if double_precision:
        recovery_model = recovery_model.double()

    if use_cuda and torch.cuda.is_available():
        recovery_model = recovery_model.cuda()

    recovery_model.eval()
    return recovery_model


def create_blackbox_model(forward, input_dim=None, output_dim=None, *, device=None,
                          dtype=None, batch_size=256, max_queries=None):
    """Wrap a Tensor query interface; a full model may supply dimensions and placement."""
    if input_dim is None:
        input_dim = forward.input_dim
    if output_dim is None:
        output_dim = forward.output_dim
    device = device if device is not None else getattr(forward, 'device', 'cuda')
    dtype = dtype if dtype is not None else getattr(forward, 'dtype', torch.float64)
    return BlackBoxDNN(forward, input_dim, output_dim, device=device, dtype=dtype,
                       batch_size=batch_size, max_queries=max_queries)
