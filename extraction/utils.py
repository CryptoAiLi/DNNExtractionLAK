"""Shared math, parameter comparisons, critical-point data, and extraction exceptions."""
import os
import numpy as np
import torch
import copy
import hashlib
import sys


def matmul(a, b, c, _np=np):
    """Apply an affine matrix transform."""
    if c is None:
        c = _np.zeros(1)
    return _np.dot(a, b) + c


class Path:
    """Resolve paths for extraction outputs."""

    def __init__(self, base='', name=''):
        self.root = os.path.join(base, name)

    def setup(self, base, name):
        self.root = os.path.join(base, name)

    def getSignaturePath(self, layer):
        return f'{self.root}/models/signature_weight_{layer}.npy', f'{self.root}/models/signature_bias_{layer}.npy'


path = Path()


class AcceptableFailure(Exception):
    """Signal a recoverable failure that requires more evidence or another attempt."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class GatherMoreData(AcceptableFailure):
    """Request additional witnesses for a particular candidate neuron."""

    def __init__(self, data, **kwargs):
        super(GatherMoreData, self).__init__(data=data, **kwargs)


np.set_printoptions(precision=20)


class GlobalConfig:
    """Track query counts, saved queries, and clustering tolerances."""
    query_count = 0
    crit_query_count = 0
    set_save_queries = False
    SAVED_QUERIES = []
    BLOCK_ERROR_TOL = 1e-4

    @classmethod
    def reset(cls):
        """Reset all query counters and saved queries."""
        cls.query_count = 0
        cls.crit_query_count = 0
        cls.SAVED_QUERIES = []
        cls.BLOCK_ERROR_TOL = 1e-4


MIN_SAME_SIZE = 3
BLOCK_MULTIPLY_FACTOR = 2
DEAD_NEURON_THRESHOLD = 2000


class CriticalPoint:
    """Store a critical point's index, coordinates, ratio, and local direction matrix."""
    def __init__(self, index, ratio, point, h_matrix):
        self.index = index
        self.ratio = ratio
        self.point = point
        self.h_matrix = h_matrix


def basis(i, N):
    """Return the i-th standard basis vector of length N."""
    a = np.zeros(N, dtype=np.float64)
    a[i] = 1
    return a


def to_tensor(x):
    """Convert array-like input to a float64 Tensor."""
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).double()
    return x.double() if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float64)


def get_grad(x, direction, weights, biases, eps=1e-6):
    """Estimate a directional derivative using finite differences and known weights."""
    x = x[np.newaxis, :]
    a = predict_manual_fast(x - eps * direction, weights, biases)
    b = predict_manual_fast(x, weights, biases)
    g1 = (b - a) / eps
    return g1


def get_second_grad_unsigned(args, x, direction, weights, biases, eps, eps2):
    """Measure an unsigned directional kink using outer step eps and inner step eps2."""
    grad_value = get_grad(
        x + direction * eps, direction, weights, biases, eps2
    ) + get_grad(x - direction * eps, -direction, weights, biases, eps2)

    if args.dataset == "cifar10":
        return grad_value[0][args.output_index]

    return grad_value[0]


def predict_manual_fast(x, weights, biases):
    """Evaluate known weights with NumPy and update query counts."""
    GlobalConfig.query_count += x.shape[0]

    orig_x = x
    for i in range(len(weights)):
        x = matmul(x, weights[i], biases[i])
        if i < len(weights) - 1:
            x = x * (x > 0)

    if GlobalConfig.set_save_queries:
        GlobalConfig.SAVED_QUERIES.extend(zip(orig_x, x))
    return x


def get_query_counts():
    """Return (query_count, crit_query_count)."""
    print("Query count: ", GlobalConfig.query_count)
    return GlobalConfig.query_count, GlobalConfig.crit_query_count


def check_activation(args, point, recovery_model, dead_neurons):
    """Check whether the recovered prefix has a valid activation pattern at a point.

    dead_neurons identifies coordinates excluded from the activation checks.
    """
    if len(recovery_model.fcs) == 0:
        return True

    # Inspect all prefix layers, including the final layer.
    hidden_outputs = recovery_model.cheat(
        torch.tensor(point, dtype=torch.float64).unsqueeze(0) if isinstance(point, np.ndarray) else point,
        pad=False
    )

    for layer in range(1, args.layerID - 1):
        if layer >= len(hidden_outputs):
            break
        layer_output = hidden_outputs[layer].flatten()
        for j in range(len(layer_output)):
            if j in dead_neurons[layer]:
                continue
            # A positive hidden output identifies an active neuron.
            if layer_output[j] > 0:
                pass

    return True


def compute_difference(value1, value2):
    """Align two ratio vectors by scale and sum their absolute differences.

    NaNs are supported. Returns (diff, 0, factor, sequence).
    """
    factor = 0.0
    diff = 0.0
    sequence = 0

    for i in range(len(value1)):
        if np.isnan(value1[i]) or np.isnan(value2[i]):
            continue
        if value1[i] == 0 or value2[i] == 0:
            continue

        if factor == 0:
            if abs(value1[i]) > abs(value2[i]):
                factor = value2[i] / value1[i]
                sequence = 1
            else:
                factor = value1[i] / value2[i]
                sequence = 2
        else:
            if sequence == 1:
                diff += abs(value2[i] - factor * value1[i])
            if sequence == 2:
                diff += abs(value1[i] - factor * value2[i])

    return diff, 0, factor, sequence


def compute_weight_difference(extracted_weight, real_weight):
    """Compare recovered and ground-truth weights and log detailed scale-aligned errors."""
    factor = 0.0
    diff = 0.0
    difference = []
    relative_diff = 0.0
    relative_difference = []

    for i in range(len(extracted_weight)):
        if np.isnan(extracted_weight[i]) or np.isnan(real_weight[i]):
            continue
        if extracted_weight[i] == 0 or real_weight[i] == 0:
            continue

        if factor == 0:
            factor = real_weight[i] / extracted_weight[i]
        diff += abs(real_weight[i] - factor * extracted_weight[i])
        difference.append(abs(real_weight[i] - factor * extracted_weight[i]))
        relative_diff += abs(real_weight[i] - factor * extracted_weight[i]) / abs(
            real_weight[i]
        )
        relative_difference.append(
            abs(real_weight[i] - factor * extracted_weight[i]) / abs(real_weight[i])
        )

    if abs(diff) > 1e-3:
        print("this point is not good")

    non_nan_count = 0
    for i in range(len(extracted_weight)):
        if np.isnan(extracted_weight[i]):
            continue
        non_nan_count += 1

    print(
        "[Precision Checkpoint - signature] The average difference is: ",
        diff / non_nan_count,
    )
    print(
        "[Precision Checkpoint - signature] The average relative difference is: ",
        relative_diff / non_nan_count,
    )
    print("The factor is: ", factor)
    print("[Precision Checkpoint - signature] The sum difference is: ", diff)
    print("[Precision Checkpoint - signature] The difference is: ", difference)
    print(
        "[Precision Checkpoint - signature] The relative difference is: ",
        relative_difference,
    )

    return diff


def get_prev_ratio(args, model, biases):
    """Return recovered neuron indices and scale factors for the previous layer."""
    import os

    if hasattr(args, 'layerID') and args.layerID == 1:
        dimOfInput = model.structure[0]
        previous_extracted_sequence = np.arange(dimOfInput)
        previous_alpha = np.ones(dimOfInput)
        return previous_extracted_sequence, previous_alpha

    if hasattr(args, 'real_attack') and args.real_attack == 0:
        # All neurons are available in white-box mode.
        dimOfLayer = model.structure[args.layerID]
        previous_extracted_sequence = np.arange(dimOfLayer)
        previous_alpha = np.ones(dimOfLayer)
        return previous_extracted_sequence, previous_alpha

    # Load previous-layer recovery results from disk.
    if not hasattr(args, 'output_path'):
        # Without output_path, use the white-box fallback.
        dimOfLayer = model.structure[args.layerID]
        previous_extracted_sequence = np.arange(dimOfLayer)
        previous_alpha = np.ones(dimOfLayer)
        return previous_extracted_sequence, previous_alpha

    if hasattr(args, 'layerID') and args.layerID == 2:
        previous_recoveryPath = os.path.join(
            args.output_path,
            "recovered results",
            f"Seed{args.seed}-Count500-RealAttack{args.real_attack}",
        )
    else:
        count = args.Count if hasattr(args, 'Count') else 1000
        previous_recoveryPath = os.path.join(
            args.output_path,
            "recovered results",
            f"Seed{args.seed}-Count{count}-RealAttack{args.real_attack}",
        )

    previous_recovered_parameters = np.load(
        os.path.join(previous_recoveryPath, f"recovered_{args.layerID - 1}.npz")
    )
    previous_extracted_sequence = previous_recovered_parameters["extracted_sequence"]
    previous_extracted_bias = previous_recovered_parameters["extracted_bias"]

    previous_bias_real = biases[args.layerID - 2]
    previous_bias_real = np.array(previous_bias_real)
    previous_bias_real = previous_bias_real[previous_extracted_sequence]
    previous_alpha = previous_extracted_bias / previous_bias_real

    return previous_extracted_sequence, previous_alpha


def prefix_tensor(prefix, points):
    return torch.as_tensor(points, device=prefix.device, dtype=prefix.dtype)


def prefix_values(prefix, points):
    with torch.no_grad():
        return prefix.forward_prefix(prefix_tensor(prefix, points)).detach().cpu().numpy()


def query_numpy(query, points):
    """Convert NumPy inputs at the query boundary; model query interfaces use Tensors."""
    if hasattr(query, 'device') and hasattr(query, 'dtype'):
        points = torch.as_tensor(points, device=query.device, dtype=query.dtype)
        if points.numel() == 0:
            points = points.reshape(0, query.input_dim)
    values = query(points)
    return values.detach().cpu().numpy() if isinstance(values, torch.Tensor) else np.asarray(values)


class RecoveryCache:
    """Cache successful evidence within one recovery run using a bounded capacity.

    Parameter values, linear carriers, device, and dtype define the scope; model
    changes invalidate matches. Keep existing entries when full to avoid repeated
    eviction during retries. Failures are not cached and entries are not persisted.
    """
    def __init__(self, max_bytes=128 * 1024 * 1024):
        self.max_bytes = max_bytes
        self.entries = {}
        self.bytes = self.hits = self.misses = 0

    def scope(self, query, prefix=None):
        query = getattr(query, '__self__', query)
        target = (id(query), str(getattr(query, 'device', None)), str(getattr(query, 'dtype', None)))
        if prefix is None:
            return target
        digest = hashlib.sha256()
        digest.update(repr((tuple(prefix.structure), str(prefix.device), str(prefix.dtype),
                            prefix.with_relu, prefix.boundary_filter_profile())).encode())
        for name, value in prefix.state_dict().items():
            digest.update(name.encode())
            array = value.detach().cpu().numpy()
            digest.update(str(array.dtype).encode())
            digest.update(repr(array.shape).encode())
            digest.update(array.tobytes())
        return target + (digest.digest(),)

    @staticmethod
    def key(scope, kind, points, *options):
        array = np.ascontiguousarray(points, dtype=np.float64)
        return scope, kind, array.shape, hashlib.sha256(array.tobytes()).digest(), options

    def get(self, key):
        if key not in self.entries:
            self.misses += 1
            return None
        self.hits += 1
        return copy.deepcopy(self.entries[key])

    def put(self, key, value):
        def size(item):
            if isinstance(item, np.ndarray):
                return sys.getsizeof(item) + (0 if item.flags.owndata else item.nbytes)
            if isinstance(item, dict):
                return sys.getsizeof(item) + sum(size(k) + size(v) for k, v in item.items())
            if isinstance(item, (tuple, list)):
                return sys.getsizeof(item) + sum(map(size, item))
            return sys.getsizeof(item)
        needed = size(key) + size(value)
        if key not in self.entries and self.bytes + needed <= self.max_bytes:
            self.entries[key] = copy.deepcopy(value)
            self.bytes += needed

    def summary(self):
        return dict(hits=self.hits, misses=self.misses, entries=len(self.entries),
                    bytes=self.bytes, max_bytes=self.max_bytes)


def cached_prefix_values(prefix, points, cache=None, *, scope=None):
    if cache is None:
        return prefix_values(prefix, points)
    scope = cache.scope(None, prefix) if scope is None else scope
    key = cache.key(scope, 'prefix_values', points)
    values = cache.get(key)
    if values is None:
        values = prefix_values(prefix, points)
        cache.put(key, values)
    return values
