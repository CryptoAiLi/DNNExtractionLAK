"""DNN interfaces for queries, white-box validation, and recovered prefixes.

BaseDNN defines public input/output operations. WhiteBoxDNN stores known
structure and parameters; RecoveryModel operates only on recovered information.
"""
from numbers import Integral
from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import nn

from extraction.utils import matmul
from .utils import get_parameters, set_parameters, prepare_parameters


class BaseDNN(nn.Module, ABC):
    """Define public input/output operations without storing internal layers."""

    def __init__(self, input_dim, output_dim, *, device='cuda', dtype=None):
        super().__init__()
        for value in (input_dim, output_dim):
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError('Dimensions must be positive integers')
        self._input_dim, self._output_dim = int(input_dim), int(output_dim)
        self.register_buffer('_placement', torch.empty(0, device=device, dtype=dtype), persistent=False)
        self.output_mode = 'hard-label'

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return self._output_dim

    @property
    def device(self):
        return self._placement.device

    @property
    def dtype(self):
        return self._placement.dtype

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError('Subclasses must implement the input/output interface')

    @torch.no_grad()
    def forward_eval(self, x):
        return self.forward(x)

    def bmodel(self, x):
        """Return argmax class indices with shape (batch_size,) and dtype int32."""
        return (self.forward_eval(x).argmax(1)).to(torch.int32)


    def gap(self, x):
        """Compute the gap between the largest and second-largest output logits."""
        return self.gapt(x)

    def gapt(self, x, grad=False):
        """Return the output margin; white-box forward may retain gradients."""
        out = self.forward(x) if grad else self.forward_eval(x)
        if out.shape[-1] < 2:
            raise ValueError('GAP requires at least two output coordinates')
        top = out.topk(2, dim=1).values
        return top[:, 0] - top[:, 1]


class QueryBudgetExceeded(RuntimeError):
    """Signal an exhausted query budget so the caller can save the recovered prefix."""


class BlackBoxDNN(BaseDNN):
    """Expose Tensor queries without registering target parameters or activations.

    forward and query share query counts and budgets. Callers convert input types,
    devices, and precision; the target loader controls target placement. Private
    fields enforce an API boundary, not process-level security isolation.
    """

    def __init__(self, forward, input_dim, output_dim, *, device='cuda',
                 dtype=torch.float64, batch_size=256, max_queries=None):
        super().__init__(input_dim, output_dim, device=device, dtype=dtype)
        if not callable(forward):
            raise TypeError('forward must be callable')
        for value in (batch_size, max_queries):
            if value is not None and (isinstance(value, bool) or not isinstance(value, Integral) or value <= 0):
                raise ValueError('Query batch size and budget must be positive integers')
        if batch_size is None:
            raise ValueError('batch_size must be a positive integer')
        # nn.Module.__setattr__ would register a target Module and expose its weights.
        object.__setattr__(self, '_query_forward', forward)
        self.batch_size, self.max_queries = int(batch_size), max_queries
        self.query_count = 0

    @torch.no_grad()
    def query(self, points: torch.Tensor) -> torch.Tensor:
        if not isinstance(points, torch.Tensor):
            raise TypeError('Query inputs must be Tensor; convert them at the call site')
        if points.device != self.device or points.dtype != self.dtype:
            raise ValueError('Query input device and dtype must match the model')
        if points.ndim not in (1, 2) or points.shape[-1] != self.input_dim:
            raise ValueError('Expected (input_dim,) or (batch, input_dim)')
        points = points.reshape(-1, self.input_dim)
        if not torch.isfinite(points).all():
            raise ValueError('Query inputs must be finite')
        if self.max_queries is not None and self.query_count + len(points) > self.max_queries:
            raise QueryBudgetExceeded(f'Query budget {self.max_queries} exhausted at {self.query_count}')
        outputs = []
        with torch.no_grad():
            for start in range(0, len(points), self.batch_size):
                batch = points[start:start + self.batch_size]
                self.query_count += len(batch)
                value = self._query_forward(batch)
                if not isinstance(value, torch.Tensor):
                    raise TypeError('Query outputs must be Tensor')
                if value.device != self.device or value.dtype != self.dtype:
                    raise ValueError('Query output device and dtype must match the model')
                if value.shape != (len(batch), self.output_dim) or not torch.isfinite(value).all():
                    raise ValueError('Oracle must return finite logits with shape (batch, output_dim)')
                outputs.append(value.detach().clone())
        return torch.cat(outputs) if outputs else points.new_empty((0, self.output_dim))

    def forward(self, points):
        return self.query(points)

    def tensor_forward(self, points):
        return self.forward(points)


class WhiteBoxDNN(BaseDNN):
    """Represent a network with known structure and parameters for internal inspection."""

    def __init__(self, structure, *, device='cuda', dtype=None, with_relu=False):
        if isinstance(structure, Integral) and not isinstance(structure, bool):
            structure = [structure]
        structure = list(structure)
        if not structure or any(isinstance(n, bool) or not isinstance(n, Integral) or n <= 0 for n in structure):
            raise ValueError('structure must contain positive integer dimensions')
        super().__init__(structure[0], structure[-1], device=device, dtype=dtype)
        self.structure = [int(n) for n in structure]
        self.fcs = nn.Sequential(*[
            nn.Linear(a, b, device=self.device, dtype=self.dtype)
            for a, b in zip(self.structure, self.structure[1:])
        ])
        self.activation = nn.ReLU()
        self.with_relu = with_relu

    @property
    def output_dim(self):
        return self.structure[-1]

    def set_weights_bias(self, weights, bias, *, layout='out_in'):
        """Set layer weights and biases after validating their shapes."""
        set_parameters(self, weights, bias, layout=layout)


    def get_weights_bias(self, layers=None, transposition=True):
        """Return NumPy weight and bias arrays for selected layers, or all layers if None.

        transposition=True returns weights in (in_features, out_features) layout;
        False preserves the native layout.
        """
        return get_parameters(self, layout='in_out' if transposition else 'out_in', layers=layers)


    @torch.no_grad
    def cheat(self, x, pad=True):
        """Return pre-ReLU activations for analysis.

        pad=True stacks hidden layers as (num_layers, batch_size, max_width).
        pad=False returns unpadded arrays for all layers, including the final layer.
        """
        o = []

        def relu(_x, dim):
            if pad:
                # Pad layer outputs to a common width before stacking.
                if _x.size(-1) < dim:
                    padding = dim - _x.size(-1)
                    _xx = torch.nn.functional.pad(_x, (0, padding))
                    _xx[:, -padding:] = 1
                    o.append(_xx)
                else:
                    o.append(_x)
            else:
                o.append(_x.clone())
            return self.activation(_x)

        x = x.view(-1, self.structure[0])
        for i, fc in enumerate(self.fcs[:-1]):
            x = relu(fc(x), self.structure[i + 1])

        if pad:
            # The padded interface returns hidden layers only.
            width = max((t.shape[-1] for t in o), default=0)
            return (torch.stack([nn.functional.pad(t, (0, width - t.shape[-1]), value=1) for t in o])
                    if o else x.new_empty((0, x.shape[0], 0)))
        # The unpadded interface returns all layers, including the final layer.
        if len(self.fcs) > 0:
            o.append(self.fcs[-1](x).cpu().numpy())
        return [t if isinstance(t, np.ndarray) else t.cpu().numpy() for t in o]


    def cluster2neuron(self, cluster_id):
        """Map a global neuron index to a layer index and an index within that layer.

        Global indices enumerate neurons in layer order.
        """
        if (isinstance(cluster_id, bool) or not isinstance(cluster_id, Integral)
                or not 0 <= cluster_id < sum(self.structure[1:])):
            raise ValueError('Cluster is outside the known structure')
        neuron_id = cluster_id

        i = 1
        while neuron_id >= self.structure[i]:
            neuron_id -= self.structure[i]
            i += 1

        return i - 1, neuron_id


    def neuron2cluster(self, layer, neuron_id):
        """Map a one-based layer number and local neuron index to a global neuron index."""
        if not 1 <= layer < len(self.structure) or not 0 <= neuron_id < self.structure[layer]:
            raise ValueError('Neuron is outside the known structure')
        return sum(self.structure[1:layer]) + neuron_id


    def on_which_hidden_layer(self, point):
        """Return the first hidden neuron whose preactivation is near zero.

        Returns (layer_id, neuron_id), or (-1, -1) if no boundary is found.
        """
        weights, biases = self.get_weights_bias(transposition=True)

        x = point
        for i in range(len(weights)):
            x = np.matmul(x, weights[i]) + biases[i]
            for j in range(len(x)):
                if abs(x[j]) < 1e-6:
                    return i, j
            if i < len(weights) - 1:
                x = x * (x > 0)

        return -1, -1


    @staticmethod
    def numpy_forward(x, weights, biases, with_relu=False, n_layers=None):
        """Evaluate all or selected known layers with NumPy weight arrays.

        Weights use (in_features, out_features) layout. with_relu selects whether
        the final evaluated layer applies ReLU; n_layers limits the prefix length.
        """
        layers = zip(weights, biases)
        if n_layers is not None:
            layers = list(layers)[:n_layers]
        for i, (w, b) in enumerate(layers):
            x = matmul(x, w, b)
            if (i < len(weights) - 1) or with_relu:
                x = x * (x > 0)
        return x


    @staticmethod
    def get_hidden_layers(x, weights, biases, flat=False):
        """Return pre-ReLU NumPy activations for every layer, including the final layer.

        flat=True concatenates the outputs into a one-dimensional array.
        """
        if len(weights) == 0:
            return []
        region = []
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = matmul(x, w, b)
            region.append(np.copy(x))
            if i < len(weights) - 1:
                x = x * (x > 0)
        if flat:
            region = np.concatenate(region, axis=0)
        return region


    @staticmethod
    def get_polytope(x, weights, biases, flat=False):
        """Return the activation-sign tuple identifying the input's local region."""
        if len(weights) == 0:
            return tuple()
        h = WhiteBoxDNN.get_hidden_layers(x, weights, biases)
        h = np.concatenate(h, axis=0)
        return tuple(np.int32(np.sign(h)))


    @staticmethod
    def get_local_matrix_and_bias(weights, biases, x0):
        """Compose a local affine map output = x @ M + b at x0.

        Weights use (in_features, out_features) layout. A single input returns M with
        shape (in, out) and b with shape (out,); batched inputs add a leading batch axis.
        """
        if len(x0.shape) < 2:
            M, b = WhiteBoxDNN.get_local_matrix_and_bias(weights, biases, np.array([x0]))
            return M[0], b[0]

        MM = []
        bb = []
        for x0i in x0:
            M = weights[0].copy()
            b = biases[0].copy()
            x = np.matmul(x0i, M) + b
            for layer_id in range(1, len(weights)):
                M_hat = weights[layer_id].copy()
                M_hat[x < 0] = 0
                x = np.matmul(x, M_hat) + biases[layer_id]
                b = np.matmul(b, M_hat) + biases[layer_id]
                M = np.matmul(M, M_hat)

            MM.append(M)
            bb.append(b)

        return np.array(MM), np.array(bb)


    @staticmethod
    def get_neuron_values(x, weights, biases):
        """Return a list of per-layer pre-ReLU activations using known NumPy weights."""
        values = [[]]
        for i in range(len(weights)):
            x = x @ weights[i] + biases[i]
            values.append(np.copy(x))
            x[x < 0] = 0.0
        return values


    def _layers(self, n_layers):
        if n_layers is None:
            return list(self.fcs)
        if isinstance(n_layers, bool) or not isinstance(n_layers, Integral) or not 0 <= n_layers <= len(self.fcs):
            raise ValueError('n_layers must be within the known network')
        return list(self.fcs)[:n_layers]


    def forward(self, x, with_relu=None, n_layers=None):
        """Evaluate known layers; with_relu explicitly controls the final activation."""
        if with_relu is None:
            with_relu = self.with_relu
        x = x.reshape(-1, self.structure[0])
        layers = self._layers(n_layers)
        for index, layer in enumerate(layers):
            x = layer(x)
            if index < len(layers) - 1 or with_relu:
                x = self.activate_layer(layer, x)
        return x


    @staticmethod
    def activation_mask(layer, values, *, inclusive=False):
        mask = values >= 0 if inclusive else values > 0
        linear = getattr(layer, 'linear_coordinates', None)
        return mask if linear is None else mask | linear


    @classmethod
    def activate_layer(cls, layer, values):
        linear = getattr(layer, 'linear_coordinates', None)
        return torch.relu(values) if linear is None else torch.where(linear, values, torch.relu(values))


    def has_linear_coordinates(self):
        return any(hasattr(layer, 'linear_coordinates') for layer in self.fcs)


    @torch.no_grad()
    def forward_eval(self, x, n_layers=None, with_relu=None):
        return self.forward(x, with_relu=with_relu, n_layers=n_layers)


    @torch.no_grad()
    def local_affine(self, point):
        """Return the local affine map at point, leaving the final layer before ReLU."""
        point = torch.as_tensor(point, device=self.device, dtype=self.dtype).reshape(-1)
        matrix = torch.eye(self.structure[0], device=self.device, dtype=self.dtype)
        bias = torch.zeros(self.structure[0], device=self.device, dtype=self.dtype)
        for index, layer in enumerate(self.fcs):
            if index:
                active = self.activation_mask(self.fcs[index - 1], point @ matrix + bias, inclusive=True)
                matrix = matrix * active
                bias = bias * active
            matrix = matrix @ layer.weight.T
            bias = bias @ layer.weight.T + layer.bias
        return matrix, bias


    def relu_around(self, x):
        """Apply the first sample's activation mask to the entire input batch."""
        mask = (x[:1] > 0).to(x.dtype) if self.with_relu else (x[:1] >= 0).to(x.dtype)
        return x * mask


    @torch.no_grad
    def forward_around(self, x, *, with_relu=None):
        """Evaluate a neighborhood with hidden activation masks fixed by the first sample.

        The final layer follows with_relu.
        """
        x = x.view(-1, self.structure[0])
        if with_relu is None:
            with_relu = self.with_relu
        if len(self.fcs) == 0: return x
        for index, layer in enumerate(self.fcs):
            x = layer(x)
            if index == len(self.fcs) - 1 and not with_relu:
                continue
            if hasattr(layer, 'linear_coordinates'):
                x = x * self.activation_mask(layer, x[:1], inclusive=not self.with_relu)
            else:
                x = self.relu_around(x)
        return x


    @torch.no_grad
    def forward_at(self, point, d_matrix, n_layers=None, *, with_relu=None):
        """Transform direction rows through the network's activation pattern at point.

        point has shape (input_dim,) or (1, input_dim); d_matrix has shape
        (n_directions, input_dim). n_layers limits the number of evaluated layers.
        Returns the locally transformed direction matrix.
        """
        layers = self._layers(n_layers)
        if with_relu is None:
            with_relu = self.with_relu
        if point.numel() != self.structure[0]:
            raise ValueError('forward_at expects one reference point')
        if len(layers) == 0: return d_matrix

        # Record each layer's activation mask during forward propagation.
        x = point.view(-1, self.structure[0])
        mask_vectors = []
        for index, layer in enumerate(layers):
            x = layer(x)
            mask_vectors.append(self.activation_mask(layer, x)
                                if index < len(layers) - 1 or with_relu else torch.ones_like(x))
            x = self.activate_layer(layer, x)

        # Transform direction rows through the recorded activation masks.
        h_matrix = d_matrix
        for i, layer in enumerate(layers):
            # Apply layer weights before masking inactive coordinates.
            h_matrix = h_matrix @ layer.weight.t() * mask_vectors[i]

        return h_matrix


class RecoveryModel(WhiteBoxDNN):
    """Store recovered information; an integer input creates an empty prefix.

    forward leaves the final layer linear by default; forward_prefix applies ReLU.
    A structure list can prebuild layers. Prefer append_layer during recovery.
    """

    def forward_around(self, x, *, with_relu=True):
        """Evaluate using fixed prefix activation masks, applying final ReLU by default."""
        return super().forward_around(x, with_relu=with_relu)

    def forward_at(self, point, d_matrix, n_layers=None, *, with_relu=True):
        """Transform directions through the recovered prefix, applying final ReLU by default."""
        return super().forward_at(point, d_matrix, n_layers, with_relu=with_relu)

    def append_layer(self, weight, bias, *, layout='out_in'):
        """Validate and append recovered parameters while preserving device, dtype, and RNG state."""
        bias_tensor = torch.as_tensor(bias)
        if bias_tensor.ndim != 1 or bias_tensor.numel() == 0:
            raise ValueError('bias must be a nonempty vector')
        output_dim = bias_tensor.numel()
        [(weight, bias)] = prepare_parameters(
            [weight], [bias], [(self.structure[-1], output_dim)],
            layout=layout, device=self.device, dtype=self.dtype,
        )
        layer = nn.Linear(self.structure[-1], output_dim, device='meta', dtype=self.dtype)
        layer.weight = nn.Parameter(weight)
        layer.bias = nn.Parameter(bias)
        self.fcs.add_module(str(len(self.fcs)), layer)
        layer.train(self.training)
        self.structure = [*self.structure, output_dim]
        return self


    def extend_last_layer_with_linear_input(self):
        """Represent [ReLU(z_R), h_previous]; carriers are not inferred neurons.

        Only the most recent candidate may be extended. Linear coordinates have
        no ReLU boundary, including when their input is negative or zero.
        """
        if not len(self.fcs) or hasattr(self.fcs[-1], 'linear_coordinates'):
            raise ValueError('Expected an ordinary final candidate layer')
        old = self.fcs[-1]
        width, incoming = old.out_features, old.in_features
        layer = nn.Linear(incoming, width + incoming, device='meta', dtype=self.dtype)
        layer.weight = nn.Parameter(torch.cat((old.weight.detach(),
            torch.eye(incoming, device=self.device, dtype=self.dtype))))
        layer.bias = nn.Parameter(torch.cat((old.bias.detach(), old.bias.new_zeros(incoming))))
        layer.register_buffer('linear_coordinates', torch.arange(width + incoming, device=self.device) >= width)
        layer.train(self.training)
        self.fcs[-1] = layer
        self.structure[-1] = width + incoming
        stored = getattr(self, '_boundary_relative_tolerances', None)
        if stored is not None:
            stored[-1] = np.concatenate((stored[-1], np.zeros(incoming)))
        return self


    def forward_prefix(self, x, n_layers=None):
        """Apply ReLU to every recovered layer; an empty prefix is the identity.

        Gradients are retained.
        """
        return self.forward(x, with_relu=True, n_layers=n_layers)


    @torch.no_grad()
    def local_wiggle_differences(self, point, eps=1e-6):
        """Evaluate local affine finite differences, including the final ReLU."""
        point = torch.as_tensor(point, device=self.device, dtype=self.dtype).reshape(-1)
        matrix, bias = self.local_affine(point)
        activate = (lambda value: self.activate_layer(self.fcs[-1], value)) if len(self.fcs) else torch.relu
        baseline = activate(point @ matrix + bias)
        perturbed = point + eps * torch.eye(self.structure[0], device=self.device, dtype=self.dtype)
        return activate(perturbed @ matrix + bias) - baseline


    def calibrate_boundary_tolerances(self, weights, biases, groups, *, multiplier=2.0,
                                      maximum=1e-6, batch_size=256,
                                      progress_mode='auto', progress_interval=5.0):
        """Calibrate relative boundary tolerances using candidate witnesses and the prefix.

        Return min(maximum, multiplier * max(abs(h @ w + b) / (norm(w) * (1 + norm(h))))).
        This is a capped empirical tolerance, not a rigorous error bound.
        Invalid witness groups receive zero tolerance; target parameters are not accessed.
        """
        from utils.progress import Progress
        weights, biases = np.asarray(weights), np.asarray(biases)
        if (weights.ndim != 2 or weights.shape[0] != self.structure[-1]
                or biases.shape != (weights.shape[1],) or len(groups) != len(biases)):
            raise ValueError('Weights, biases and groups must agree with prefix dimensions')
        if not np.isfinite(multiplier) or multiplier <= 0 or not np.isfinite(maximum) or maximum < 0:
            raise ValueError('Expected positive multiplier and nonnegative finite maximum')
        if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size <= 0:
            raise ValueError('batch_size must be a positive integer')
        relative = np.zeros(len(biases))
        with Progress('calibrate recovered boundary error', len(groups), progress_interval, mode=progress_mode) as bar:
            for neuron, group in enumerate(groups):
                weight, bias = weights[:, neuron], biases[neuron]
                norm = np.linalg.norm(weight)
                valid = np.isfinite(norm) and norm > 0 and np.isfinite(bias) and len(group) > 0
                if valid:
                    max_error = 0.0
                    with torch.no_grad():
                        for start in range(0, len(group), batch_size):
                            inputs = torch.as_tensor(group[start:start + batch_size], device=self.device, dtype=self.dtype)
                            hidden = self.forward_prefix(inputs).detach().cpu().numpy()
                            residual = np.abs(hidden @ (weight / norm) + bias / norm) / (1 + np.linalg.norm(hidden, axis=1))
                            if not np.isfinite(residual).all():
                                valid = False
                                break
                            max_error = max(max_error, float(residual.max()))
                    if valid:
                        relative[neuron] = min(maximum, multiplier * max_error)
                bar.update(neuron + 1, f'layer={len(self.fcs) + 1} calibrated={neuron + 1} skipped_current={not valid}')
        return relative


    def append_boundary_tolerances(self, relative):
        """Store filtering evidence for the newly appended layer separately from state_dict."""
        relative = np.asarray(relative, dtype=float)
        if (not len(self.fcs) or relative.shape != (self.fcs[-1].out_features,)
                or not np.isfinite(relative).all() or np.any(relative < 0)):
            raise ValueError('Expected finite nonnegative tolerances for the appended layer')
        previous = getattr(self, '_boundary_relative_tolerances', None)
        if previous is None:
            previous = [np.zeros(fc.out_features) for fc in self.fcs[:-1]]
        if len(previous) != len(self.fcs) - 1:
            raise ValueError('Boundary profiles must precede the newly appended layer')
        self._boundary_relative_tolerances = [np.array(r, copy=True) for r in previous] + [relative.copy()]


    def boundary_filter_profile(self):
        """Return JSON-compatible per-layer tolerances, or an empty list without calibration."""
        return [np.asarray(r).tolist() for r in getattr(self, '_boundary_relative_tolerances', [])]


    @torch.no_grad()
    def prefix_boundary_mask(self, x, *, tolerance=1e-5, n_layers=None,
                             start_layer=0, ignored_neurons=None, normalize=True,
                             relative_tolerances=None):
        """Return a (batch,) boolean Tensor marking proximity to prefix boundaries.

        Accept a single input or a batch. An empty prefix returns False. The final
        selected layer is checked regardless of with_relu. n_layers limits the prefix;
        exclude the affine output layer for a full network. start_layer sets the first
        checked layer while earlier layers still propagate normally.
        ignored_neurons maps zero-based layer indices to local indices excluded only
        from boundary checks. normalize=True uses abs(z) < norm(w) * (tolerance +
        relative * (1 + norm(h))). Relative tolerances use saved calibration by default,
        with zero for uncalibrated layers. normalize=False uses abs(z) < tolerance.
        Zero weights define no locatable boundary in normalized mode. No active-dimension
        or target-layer filter is applied; callers validate nonfinite inputs.
        """
        if isinstance(tolerance, bool) or not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError('tolerance must be positive and finite')
        layers = self._layers(n_layers)
        if normalize and relative_tolerances is None:
            stored = getattr(self, '_boundary_relative_tolerances', [])
            relative_tolerances = [stored[i] if i < len(stored) else np.zeros(layer.out_features)
                                   for i, layer in enumerate(layers)]
        if relative_tolerances is not None and not normalize:
            raise ValueError('relative_tolerances requires normalize=True')
        if relative_tolerances is not None and len(relative_tolerances) != len(layers):
            raise ValueError('Expected one relative tolerance vector per selected layer')
        relative = []
        for layer_id, layer in enumerate(layers if normalize else []):
            value = (torch.zeros(layer.out_features, device=self.device, dtype=self.dtype)
                     if relative_tolerances is None else
                     torch.as_tensor(relative_tolerances[layer_id], device=self.device, dtype=self.dtype))
            if value.shape != (layer.out_features,) or not torch.isfinite(value).all() or (value < 0).any():
                raise ValueError('Relative tolerances must be finite nonnegative vectors matching layer widths')
            relative.append(value)
        if isinstance(start_layer, bool) or not isinstance(start_layer, Integral) or not 0 <= start_layer <= len(layers):
            raise ValueError('start_layer must be within the selected prefix')
        ignored = {}
        for layer_id, indices in (ignored_neurons or {}).items():
            if isinstance(layer_id, bool) or not isinstance(layer_id, Integral) or not 0 <= layer_id < len(layers):
                raise ValueError('ignored neuron layer is outside the selected prefix')
            indices = list(indices)
            if any(isinstance(i, bool) or not isinstance(i, Integral) or not 0 <= i < layers[layer_id].out_features for i in indices):
                raise ValueError('ignored neuron index is outside the layer')
            ignored[layer_id] = indices
        x = torch.as_tensor(x, device=self.device, dtype=self.dtype)
        if x.ndim not in (1, 2) or x.shape[-1] != self.structure[0]:
            raise ValueError('Expected (input_dim,) or (batch, input_dim)')
        x = x.reshape(-1, self.structure[0])
        boundary = torch.zeros(x.shape[0], device=self.device, dtype=torch.bool)
        for layer_id, layer in enumerate(layers):
            values = layer(x)
            if layer_id >= start_layer:
                near_zero = values.abs() < tolerance
                if normalize:
                    norms = torch.linalg.vector_norm(layer.weight, dim=1)
                    bound = norms * (tolerance + relative[layer_id] *
                                     (1 + torch.linalg.vector_norm(x, dim=1, keepdim=True)))
                    near_zero = (values.abs() < bound) & (norms > 0) & torch.isfinite(bound) & torch.isfinite(values)
                if layer_id in ignored:
                    near_zero[:, ignored[layer_id]] = False
                if hasattr(layer, 'linear_coordinates'):
                    near_zero[:, layer.linear_coordinates] = False
                boundary |= near_zero.any(dim=1)
            x = self.activate_layer(layer, values)
        return boundary


    @torch.no_grad()
    def filter_prefix_points(self, x, *, min_active=None, check_boundary=True,
                             tolerance=1e-5, n_layers=None, start_layer=0,
                             ignored_neurons=None, return_mask=False):
        """Filter prefix boundaries and inputs with too few nonzero prefix coordinates.

        Return (points, n_removed), with points a 2D Tensor on the model device and dtype.
        return_mask also returns the keep mask. min_active=None disables dimension
        filtering; 2 requires at least two active coordinates. Activation counts follow
        forward_eval's with_relu setting; an empty prefix does not filter dimensions.
        check_boundary=False only filters active coordinates. Nonfinite handling is unchanged.
        """
        if min_active is not None and (
            isinstance(min_active, bool) or not isinstance(min_active, Integral) or min_active < 0
        ):
            raise ValueError('min_active must be a nonnegative integer or None')
        layers = self._layers(n_layers)
        points = torch.as_tensor(x, device=self.device, dtype=self.dtype)
        if points.ndim not in (1, 2) or points.shape[-1] != self.structure[0]:
            raise ValueError('Expected (input_dim,) or (batch, input_dim)')
        points = points.reshape(-1, self.structure[0])
        keep = torch.ones(points.shape[0], device=self.device, dtype=torch.bool)
        if check_boundary:
            keep &= ~self.prefix_boundary_mask(
                points, tolerance=tolerance, n_layers=n_layers,
                start_layer=start_layer, ignored_neurons=ignored_neurons)
        if min_active is not None and layers:
            values = self.forward_eval(points, n_layers=n_layers)
            keep &= torch.count_nonzero(values, dim=1) >= min_active
        result = (points[keep], int((~keep).sum().item()))
        return (*result, keep) if return_mask else result


    @torch.no_grad()
    def filter_prefix_points_batched(self, x, *, batch_size=1024, min_active=None,
                                     check_boundary=True, tolerance=1e-5,
                                     n_layers=None, start_layer=0,
                                     ignored_neurons=None, return_mask=False):
        """Apply filter_prefix_points in batches while preserving input order.

        Accept a single point or a 2D Tensor, array, or list. Return the full filtered
        Tensor on the model device, removal count, and optionally the keep mask.
        CPU inputs move to the device per batch. RNG state and filter semantics are unchanged.
        """
        if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size <= 0:
            raise ValueError('batch_size must be a positive integer')
        points = x if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=self.dtype)
        if points.ndim == 1 and points.numel() == 0:
            points = points.reshape(0, self.structure[0])
        if points.ndim not in (1, 2) or points.shape[-1] != self.structure[0]:
            raise ValueError('Expected (input_dim,) or (batch, input_dim)')
        points = points.reshape(-1, self.structure[0])
        options = dict(min_active=min_active, check_boundary=check_boundary,
                       tolerance=tolerance, n_layers=n_layers, start_layer=start_layer,
                       ignored_neurons=ignored_neurons, return_mask=return_mask)
        if len(points) == 0:
            return self.filter_prefix_points(points, **options)
        filtered, masks, removed = [], [], 0
        for start in range(0, len(points), batch_size):
            result = self.filter_prefix_points(points[start:start + batch_size], **options)
            filtered.append(result[0])
            removed += result[1]
            if return_mask:
                masks.append(result[2])
        result = (torch.cat(filtered, dim=0), removed)
        return (*result, torch.cat(masks)) if return_mask else result


    @torch.no_grad()
    def filter_previous_layer_points(self, points, *, min_active=2, batch_size=1024):
        """Return (point_list, removal_count) after batched prefix filtering.

        Preserve order and the original point objects. min_active defaults to 2 and
        None disables dimension filtering. An empty prefix retains every point.
        The boundary tolerance is 1e-5.
        """
        points = list(points)
        batch = (torch.stack(points) if points and isinstance(points[0], torch.Tensor)
                 else np.asarray(points))
        _, n_removed, keep = self.filter_prefix_points_batched(
            batch, min_active=min_active, batch_size=batch_size, return_mask=True)
        return [point for point, retained in zip(points, keep.cpu().tolist()) if retained], n_removed


    @torch.no_grad()
    def cheat(self, x, pad=True):
        if not self.with_relu and not self.has_linear_coordinates():
            return super().cheat(x, pad=pad)
        x = x.reshape(-1, self.structure[0])
        values = []
        for layer in self.fcs:
            value = layer(x)
            values.append(value)
            x = self.activate_layer(layer, value)
        if not pad:
            return [value.cpu().numpy().copy() for value in values]
        if not values:
            return x.new_empty((0, x.shape[0], 0))
        width = max(value.shape[-1] for value in values)
        return torch.stack([nn.functional.pad(value, (0, width - value.shape[-1]), value=1) for value in values])


    def addLayers(self, layers):
        """Append an initialized Linear layer connected to the current output.

        layers is the new output width. Prefer append_layer for recovered parameters.
        """
        if isinstance(layers, bool) or not isinstance(layers, Integral) or layers <= 0:
            raise ValueError('layers must be a positive integer')
        layer = nn.Linear(self.structure[-1], int(layers), device=self.device, dtype=self.dtype)
        self.fcs.add_module(str(len(self.fcs)), layer)
        layer.train(self.training)
        self.structure = [*self.structure, int(layers)]


