"""Detect extraction errors using output queries and rank consistency.

Directional kink measurements must lie in the column space of the local prefix
matrix: rank(H) == rank([H | ys]). The target supplies logits; the prefix supplies
local transformations. Keep these roles separate when evaluating missing neurons.
"""
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import torch


# The stencil cancels affine terms and isolates kink contributions.
MASK = np.array([1, -1, 1, -1])

DEFAULT_EPS = 1e-5
DEFAULT_ORIENTATION_TOL = 1e-4
DEFAULT_RANK_TOL = 1e-4
DEFAULT_DIRECTION_FACTOR = 2
ZERO_COLUMN_TOL = 1e-8


def _placement(module, device=None, dtype=None):
    """Resolve device and dtype from overrides, module attributes, then parameters.

    Fallbacks are CUDA and float64.
    """
    module = getattr(module, '__self__', module)
    if device is None:
        device = getattr(module, 'device', None)
    if dtype is None:
        dtype = getattr(module, 'dtype', None)
    if device is None or dtype is None:
        parameters = getattr(module, 'parameters', None)
        reference = next(parameters(), None) if callable(parameters) else None
        if reference is not None:
            device = reference.device if device is None else device
            dtype = reference.dtype if dtype is None else dtype
    return (
        torch.device('cuda') if device is None else torch.device(device),
        torch.float64 if dtype is None else dtype,
    )


def second_grad_unsigned(
    model,
    point,
    direction,
    eps: float = DEFAULT_EPS,
    eps2: Optional[float] = None,
    *,
    out_index: int = 0,
    device=None,
    dtype=None,
) -> float:
    """Measure an unsigned directional kink in the selected output logit.

    Query point +/- (eps - eps2) * direction and point +/- eps * direction, then
    combine the values with [1, -1, 1, -1] and divide by eps. For a smooth function,
    the leading term is ((eps - eps2)**2 - eps**2) * g''(0) / eps; at a ReLU kink,
    the measurement is proportional to abs(w @ direction).
    The callable accepts (batch, input_dim) and returns (batch, n_logits).
    eps2 defaults to eps / 3. Device and dtype overrides take precedence.
    Returns a scalar measurement with the stencil's common scale factor.
    """
    point = np.asarray(point, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    if point.ndim != 1:
        raise ValueError(f'point must be a 1D array; got shape={point.shape}')
    if direction.shape != point.shape:
        raise ValueError(f'direction shape={direction.shape} does not match point shape={point.shape}')
    if eps2 is None:
        eps2 = eps / 3

    query_device, query_dtype = _placement(model, device, dtype)
    offsets = np.stack([
        point + direction * (eps - eps2),
        point + direction * eps,
        point - direction * (eps - eps2),
        point - direction * eps,
    ])
    offsets = torch.as_tensor(offsets, dtype=query_dtype, device=query_device)
    with torch.no_grad():
        output = model(offsets)
    output = output.detach().to('cpu').numpy()
    if output.ndim != 2:
        raise ValueError(f'Model output must have shape (batch, n_logits); got shape={output.shape}')
    if not 0 <= out_index < output.shape[1]:
        raise ValueError(f'out_index={out_index} is outside the {output.shape[1]} output columns')
    return float(np.dot(output[:, out_index].flatten(), MASK) / eps)


def measure_second_derivatives(
    model,
    point,
    directions,
    *,
    eps: float = DEFAULT_EPS,
    eps2: Optional[float] = None,
    tolerance: float = DEFAULT_ORIENTATION_TOL,
    out_index: int = 0,
    device=None,
    dtype=None,
    direction_batch_size: int = 32,
):
    """Measure and orient directional kink values relative to the first direction.

    Compare abs(ys[0] + value) / 2 and abs(ys[0] - value) / 2 with the measurement
    along (direction + reference) / 2. The smaller error determines the sign.
    If both errors exceed the absolute tolerance, return the processed values
    with anomaly=True; the caller counts the point as skipped.
    Queries are batched in direction order. An anomaly stops later batches,
    although the current batch may already contain extra queries. The tolerance
    depends on the stencil scale; batching may introduce floating-point differences.
    """
    if isinstance(direction_batch_size, bool) or not isinstance(direction_batch_size, (int, np.integer)) or direction_batch_size <= 0:
        raise ValueError('direction_batch_size must be a positive integer')
    if eps2 is None:
        eps2 = eps / 3
    point = np.asarray(point, dtype=np.float64)
    directions = np.asarray(list(directions), dtype=np.float64)
    if not len(directions):
        return np.empty(0, dtype=np.float64), False
    if point.ndim != 1 or directions.ndim != 2 or directions.shape[1:] != point.shape:
        raise ValueError('Expected point (dim,) and directions (n, dim)')
    query_device, query_dtype = _placement(model, device, dtype)

    def measure(batch):
        # Preserve the four stencil coordinates and their order. Bound memory by
        # batching directions, rather than allocating all query inputs at once.
        offsets = np.stack([
            point + batch * (eps - eps2), point + batch * eps,
            point - batch * (eps - eps2), point - batch * eps,
        ], axis=1).reshape(-1, len(point))
        with torch.no_grad():
            output = model(torch.as_tensor(offsets, dtype=query_dtype, device=query_device))
        if output.ndim != 2 or output.shape[0] != len(offsets):
            raise ValueError('Model output must have shape (batch, n_logits)')
        if not 0 <= out_index < output.shape[1]:
            raise ValueError(f'out_index={out_index} outside model output columns')
        logits = output[:, out_index].detach().cpu().numpy().reshape(-1, 4)
        return np.array([np.dot(row, MASK) / eps for row in logits])

    reference = directions[0]
    values = [float(measure(directions[:1])[0])]
    for start in range(1, len(directions), direction_batch_size):
        batch = directions[start:start + direction_batch_size]
        paired = np.stack((batch, (batch + reference) / 2), axis=1).reshape(-1, len(point))
        measured = measure(paired).reshape(-1, 2)
        for value, both in measured:
            positive_error = abs(abs(values[0] + value) / 2 - abs(both))
            negative_error = abs(abs(values[0] - value) / 2 - abs(both))
            if positive_error > tolerance and negative_error > tolerance:
                return np.asarray(values, dtype=np.float64), True
            if negative_error < positive_error:
                value = -value
            values.append(value)
    return np.asarray(values, dtype=np.float64), False


def count_directions(
    prefix,
    point,
    *,
    direction_factor: int = DEFAULT_DIRECTION_FACTOR,
) -> int:
    """Count directions as direction_factor times the nonzero final preactivations.

    Use with_relu=False regardless of the prefix's default activation setting.
    """
    prefix_device, prefix_dtype = _placement(prefix)
    point = np.asarray(point, dtype=np.float64).reshape(1, -1)
    batch = torch.as_tensor(point, dtype=prefix_dtype, device=prefix_device)
    with torch.no_grad():
        preactivation = prefix.forward(batch, with_relu=False)
    nonzero = int(np.count_nonzero(preactivation.detach().to('cpu').numpy()))
    return nonzero * int(direction_factor)


def _numerical_rank(singular_values, shape, rank_tol) -> int:
    """Compute rank from singular values.

    rank_tol=None uses s_max * max(shape) * machine_epsilon, matching matrix_rank.
    Otherwise, rank_tol is an absolute singular-value threshold.
    """
    if singular_values.size == 0:
        return 0
    if rank_tol is None:
        cutoff = singular_values[0] * max(shape) * np.finfo(singular_values.dtype).eps
    else:
        cutoff = float(rank_tol)
    return int(np.count_nonzero(singular_values > cutoff))


def rank_consistency(h_matrix, ys, *, rank_tol: Optional[float] = DEFAULT_RANK_TOL):
    """Compare the ranks of h_matrix and [h_matrix | ys].

    Returns (rank, rank_augmented, consistent).
    """
    h_matrix = np.asarray(h_matrix, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64).reshape(-1)
    if h_matrix.ndim != 2:
        raise ValueError(f'h_matrix must be a 2D array; got shape={h_matrix.shape}')
    if h_matrix.shape[0] != ys.shape[0]:
        raise ValueError(f'ys length {ys.shape[0]} does not match h_matrix row count {h_matrix.shape[0]}')
    augmented = np.column_stack((h_matrix, ys))
    rank = _numerical_rank(np.linalg.svd(h_matrix, compute_uv=False), h_matrix.shape, rank_tol)
    rank_augmented = _numerical_rank(
        np.linalg.svd(augmented, compute_uv=False), augmented.shape, rank_tol
    )
    return rank, rank_augmented, rank == rank_augmented


@dataclass
class PointReport:
    """Store a single-point rank-consistency result.

    consistent is None when an orientation anomaly prevents a decision.
    rank and rank_augmented describe H and [H | ys]; direction_count records
    the sampled directions. ys and h_matrix retain the oriented measurements
    and local prefix matrix. zero_columns is diagnostic and does not affect the decision.
    """

    consistent: Optional[bool]
    rank: int
    rank_augmented: int
    direction_count: int
    anomaly: bool = False
    ys: Optional[np.ndarray] = None
    h_matrix: Optional[np.ndarray] = None
    zero_columns: Optional[np.ndarray] = None


def check_point(
    model,
    prefix,
    point,
    *,
    eps: float = DEFAULT_EPS,
    eps2: Optional[float] = None,
    direction_factor: int = DEFAULT_DIRECTION_FACTOR,
    tolerance: float = DEFAULT_ORIENTATION_TOL,
    rank_tol: Optional[float] = DEFAULT_RANK_TOL,
    rng=None,
    out_index: int = 0,
    device=None,
    dtype=None,
) -> PointReport:
    """Check rank consistency at one critical point.

    The target supplies directional measurements; the prefix supplies the local
    matrix and direction count. rng defaults to np.random and must provide normal.
    eps2 defaults to eps / 3; out_index selects the logit. tolerance is an absolute
    orientation threshold. rank_tol is absolute unless None selects matrix_rank's
    relative threshold. Device and dtype may be overridden.
    Returns a PointReport.
    """
    point = np.asarray(point, dtype=np.float64).reshape(-1)
    direction_count = count_directions(prefix, point, direction_factor=direction_factor)
    if direction_count <= 0:
        raise ValueError('Cannot construct query directions without nonzero prefix preactivations at this point')

    generator = np.random if rng is None else rng
    directions = np.array(
        [np.sign(generator.normal(0, 1, point.shape)) for _ in range(direction_count)]
    )
    ys, anomaly = measure_second_derivatives(
        model, point, directions,
        eps=eps, eps2=eps2, tolerance=tolerance,
        out_index=out_index, device=device, dtype=dtype,
    )
    if anomaly:
        return PointReport(
            consistent=None, rank=0, rank_augmented=0,
            direction_count=direction_count, anomaly=True, ys=ys,
        )

    prefix_device, prefix_dtype = _placement(prefix)
    h_matrix = prefix.forward_at(
        torch.as_tensor(point, dtype=prefix_dtype, device=prefix_device),
        torch.as_tensor(directions, dtype=prefix_dtype, device=prefix_device),
    ).detach().to('cpu').numpy()

    zero_columns = np.mean(np.abs(h_matrix) < ZERO_COLUMN_TOL, axis=0) > 0.5
    rank, rank_augmented, consistent = rank_consistency(h_matrix, ys, rank_tol=rank_tol)
    return PointReport(
        consistent=consistent, rank=rank, rank_augmented=rank_augmented,
        direction_count=direction_count, ys=ys, h_matrix=h_matrix, zero_columns=zero_columns,
    )


@dataclass
class DetectionSummary:
    """Summarize rank checks over critical points or affine systems.

    total counts supplied units; processed counts visited units. skipped counts
    unusable measurements and inconsistent counts rank mismatches. stopped_early
    marks early termination; reports are retained only when requested.
    """

    total: int
    processed: int = 0
    skipped: int = 0
    inconsistent: int = 0
    stopped_early: bool = False
    reports: list = field(default_factory=list)

    @property
    def tested(self) -> int:
        """Return the number of tested units: processed minus skipped."""
        return self.processed - self.skipped

    @property
    def specificity(self) -> Optional[float]:
        """Return 1 - inconsistent / tested, or None when no units were tested."""
        if self.tested <= 0:
            return None
        return 1.0 - self.inconsistent / self.tested


def check_linear_output_system(model, prefix, points, *, out_index=0,
                               rank_tol=DEFAULT_RANK_TOL, cache=None) -> PointReport:
    """Check the final hidden prefix using rank([ReLU(prefix(x)), 1]) == rank([H | y]).

    Inputs are random points rather than critical points. Only target logits are
    queried. Each system needs more equations than prefix width plus bias;
    nonfinite measurements skip the entire system. Finite sampling may miss
    inactive, redundant, or logit-independent missing neurons.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or 0 in points.shape:
        raise ValueError('points must be a nonempty (n, input_dim) array')
    skipped = PointReport(None, 0, 0, len(points), anomaly=True)
    if not np.isfinite(points).all():
        return skipped
    key = cache.key(cache.scope(model, prefix), 'linear_output', points) if cache is not None else None
    prepared = cache.get(key) if cache is not None else None
    if prepared is None:
        model_device, model_dtype = _placement(model)
        prefix_device, prefix_dtype = _placement(prefix)
        with torch.no_grad():
            activations = prefix.forward(
                torch.as_tensor(points, device=prefix_device, dtype=prefix_dtype), with_relu=True
            ).detach().cpu().numpy()
            outputs = model(torch.as_tensor(points, device=model_device, dtype=model_dtype)).detach().cpu().numpy()
        if cache is not None:
            cache.put(key, (activations, outputs))
    else:
        activations, outputs = prepared
    if activations.ndim != 2 or activations.shape[0] != len(points):
        raise ValueError('Prefix output must have shape (n, width)')
    if outputs.ndim != 2 or outputs.shape[0] != len(points) or not 0 <= out_index < outputs.shape[1]:
        raise ValueError('Invalid model output shape or out_index')
    if len(points) <= activations.shape[1] + 1:
        raise ValueError('The number of equations must exceed prefix width + 1 (including bias)')
    h_matrix = np.column_stack((activations, np.ones(len(points))))
    ys = outputs[:, out_index]
    if not np.isfinite(h_matrix).all() or not np.isfinite(ys).all():
        return skipped
    rank_key = (key, 'rank', rank_tol)
    saved_rank = cache.get(rank_key) if cache is not None else None
    if saved_rank is None:
        rank = _numerical_rank(np.linalg.svd(h_matrix, compute_uv=False), h_matrix.shape, rank_tol)
        if cache is not None:
            cache.put(rank_key, rank)
    else:
        rank = saved_rank
    augmented_matrix = np.column_stack((h_matrix, ys))
    augmented = _numerical_rank(np.linalg.svd(augmented_matrix, compute_uv=False),
                                augmented_matrix.shape, rank_tol)
    consistent = rank == augmented
    return PointReport(consistent, rank, augmented, len(points), ys=ys, h_matrix=h_matrix)


def check_linear_output_systems(model, prefix, systems, *, progress=None, **kwargs):
    """Check each affine system and update progress once per system.

    DetectionSummary counts systems rather than individual input points.
    """
    systems = np.asarray(systems, dtype=np.float64)
    if systems.ndim != 3 or 0 in systems.shape[1:]:
        raise ValueError('systems must have shape (system_count, points_per_system, input_dim)')
    summary = DetectionSummary(total=len(systems))
    for points in systems:
        report = check_linear_output_system(model, prefix, points, **kwargs)
        summary.processed += 1
        summary.skipped += int(report.anomaly)
        summary.inconsistent += int(report.consistent is False)
        if progress is not None:
            progress(summary)
    return summary


def check_points(
    model,
    prefix,
    points,
    *,
    keep_reports: bool = False,
    stop_on_inconsistent: bool = False,
    verbose: bool = False,
    progress=None,
    **kwargs,
) -> DetectionSummary:
    """Check critical points and aggregate their PointReports.

    keep_reports retains per-point results and may use substantial memory.
    stop_on_inconsistent stops at the first mismatch and sets stopped_early.
    progress(summary) runs after every processed point, including skipped points;
    the callback must not mutate the summary. Other options pass to check_point.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] == 0:
        raise ValueError(f'points must be a 2D (n, dim) array; got shape={points.shape}')
    summary = DetectionSummary(total=len(points))
    warned_nonfinite = False
    for index, point in enumerate(points):
        if not np.isfinite(point).all():
            summary.processed += 1
            summary.skipped += 1
            if verbose and not warned_nonfinite:
                print(f'  Skipping nonfinite critical points (first at index {index})')
                warned_nonfinite = True
            if progress is not None:
                progress(summary)
            continue
        report = check_point(model, prefix, point, **kwargs)
        summary.processed += 1
        if report.anomaly:
            summary.skipped += 1
        elif report.consistent is False:
            summary.inconsistent += 1
        if keep_reports:
            summary.reports.append(report)
        if stop_on_inconsistent and report.consistent is False:
            summary.stopped_early = True
        if progress is not None:
            progress(summary)
        if summary.stopped_early:
            break
    return summary
