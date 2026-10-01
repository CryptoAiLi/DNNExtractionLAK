"""Improve constraint precision and refine recovered signatures."""
from extraction.utils import query_numpy
import itertools
from itertools import combinations
import math
import numpy as np
from argparse import Namespace
from scipy.linalg import null_space
from extraction.utils import prefix_values, cached_prefix_values
from utils.progress import Progress
import torch
from utils.progress import log_progress
from extraction.utils import prefix_tensor


# Licensed under MIT.

def find_diverse_subsets(data, n):
    """data: List of binary vectors (list of lists or np.array), shape: N x m

    n: target subset size
    return: list of index-tuples of subsets whose bitwise OR is all-1
    """
    N = len(data)
    m = len(data[0])
    data = np.array(data)
    result = []
    for indices in itertools.combinations(range(N), n):
        subset = data[list(indices)]
        bitwise_or = np.bitwise_or.reduce(subset, axis=0)
        if np.all(bitwise_or == 1):
            result.append(indices)
        if len(result) > 1000:
            break
    return result

def find_good_rank_matrices(stacked_input_full, n):
    """stacked_input_full: the whole matrix of stacked inputs, shape: N x m

    n: target rank of the submatrix
    """
    N, m = stacked_input_full.shape
    upper_index = 0
    lower_index = 0
    for i in range(1, N):
        submatrix = stacked_input_full[:i, :]
        if np.linalg.matrix_rank(submatrix) == n:
            upper_index = i - 1
            break
    for i in range(N - 1, 0, -1):
        submatrix = stacked_input_full[i:, :]
        if np.linalg.matrix_rank(submatrix) == n:
            lower_index = i + 1
            break
    return [stacked_input_full[:upper_index, :], stacked_input_full[lower_index:, :]]

def brute_force_max_row_subset_with_rank(stacked_input_full, target_rank):
    n_rows = stacked_input_full.shape[0]
    max_subset = []
    for subset_size in reversed(range(0, n_rows + 1)):
        count = math.comb(n_rows, subset_size)
        if count > 10000000.0:
            break
        for row_indices in combinations(range(n_rows), subset_size):
            submatrix = stacked_input_full[list(row_indices), :]
            rank = np.linalg.matrix_rank(submatrix)
            if rank == target_rank:
                return (submatrix, list(row_indices))
    return (None, [])

def find_linearly_independent_rows(args, stacked_input_full):
    """stacked_input_full: the whole matrix of stacked inputs, shape: N x m

    return: indices of linearly independent rows
    """
    independent_rows = []
    n = stacked_input_full.shape[0]
    m = stacked_input_full.shape[1]
    random_indices = args.rng.choice(np.arange(0, n), size=m, replace=False)
    basic_submatrix = stacked_input_full[random_indices, :]
    U, S, Vh = np.linalg.svd(basic_submatrix)
    try_count = 0
    while S[-1] > 1e-10 or S[-2] < 0.0001:
        if try_count > 100:
            return independent_rows
        random_indices = args.rng.choice(np.arange(0, n), size=m, replace=False)
        basic_submatrix = stacked_input_full[random_indices, :]
        U, S, Vh = np.linalg.svd(basic_submatrix)
        try_count += 1
    if args.layerID == 1:
        if args.dataset == 'mnist':
            step = 20
        if args.dataset == 'cifar10':
            step = 100
    else:
        step = 1
    submatrix = basic_submatrix.copy()
    step_num = 0
    for i in range(0, stacked_input_full.shape[0], step):
        submatrix = basic_submatrix.copy()
        if args.layerID == 1:
            rank = np.linalg.matrix_rank(basic_submatrix)
        else:
            rank = np.linalg.matrix_rank(basic_submatrix, tol=1e-10)
        upper_bound = min(i + step, stacked_input_full.shape[0])
        for k in range(i, upper_bound):
            if k not in random_indices:
                submatrix = np.vstack((submatrix, stacked_input_full[k, :]))
        if args.layerID == 1:
            new_rank = np.linalg.matrix_rank(submatrix)
        else:
            new_rank = np.linalg.matrix_rank(submatrix, tol=1e-10)
        if new_rank > rank:
            if args.dataset == 'mnist':
                for j in range(i, upper_bound):
                    if j not in random_indices:
                        subsubmatrix = np.vstack((basic_submatrix, stacked_input_full[j, :]))
                        if args.layerID == 1:
                            subsubmatrix_rank = np.linalg.matrix_rank(subsubmatrix)
                        else:
                            subsubmatrix_rank = np.linalg.matrix_rank(subsubmatrix, tol=1e-10)
                        if subsubmatrix_rank > rank:
                            independent_rows.append(j)
                        else:
                            basic_submatrix = subsubmatrix.copy()
            if args.dataset == 'cifar10':
                for j in range(i, upper_bound):
                    if j not in random_indices:
                        independent_rows.append(j)
        else:
            basic_submatrix = submatrix.copy()
    return independent_rows


def improve_precision(prefix, weights, biases, groups, *, tolerance=1e-5, rng=None,
                      progress_mode='auto', progress_interval=5.0, cache=None):
    weights, biases = weights.copy(), biases.copy()
    improved = np.zeros(len(biases), dtype=bool)
    rng = rng if rng is not None else np.random.default_rng(0)
    first_layer = not len(prefix.fcs)
    args = Namespace(layerID=len(prefix.fcs) + 1, dataset='cifar10', rng=rng)
    scope = cache.scope(None, prefix) if cache is not None else None

    def decompose(matrix, tolerance, kind):
        key = cache.key(scope, kind, matrix, tolerance) if cache is not None else None
        result = cache.get(key) if cache is not None else None
        if result is None:
            result = (np.linalg.matrix_rank(matrix, tol=tolerance) if kind == 'precision_rank'
                      else null_space(matrix, rcond=tolerance))
            if cache is not None:
                cache.put(key, result)
        return result

    def select_vector(vectors, old):
        # Check near-zero coordinates at 1e-2 and require consistent scale ratios.
        for candidate in vectors.T:
            candidate = candidate.copy()
            zeros = old == 0
            if np.any(np.abs(candidate[:-1][zeros]) > 1e-2):
                continue
            candidate[:-1][zeros] = 0
            common = (old != 0) & (candidate[:-1] != 0)
            if not np.any(common):
                continue
            pivot = np.flatnonzero(common)[0]
            factor = candidate[pivot] / old[pivot]
            if np.sum(np.abs(candidate[:-1][common] - factor * old[common])) < 1e-2:
                if candidate[0] < 0:
                    candidate *= -1
                return candidate
        return None

    with Progress(f'layer={len(prefix.fcs) + 1} precision', len(groups), progress_interval,
                  mode=progress_mode) as progress:
        for index, group in enumerate(groups):
            old = weights[:, index]
            if not len(group) or not np.all(np.isfinite(old)):
                progress.update(index + 1, 'skipped: missing direction or points')
                continue
            columns = np.flatnonzero(old != 0) if first_layer else np.arange(len(old))
            if not len(columns):
                progress.update(index + 1, 'skipped: zero direction')
                continue
            hidden = cached_prefix_values(prefix, group, cache, scope=scope)[:, columns]
            design = np.column_stack((hidden, np.ones(len(hidden))))
            rank_tol = None if first_layer else tolerance
            rank = decompose(design, rank_tol, 'precision_rank')
            if rank == design.shape[1]:
                rejected = find_linearly_independent_rows(args, design)
                design = np.delete(design, rejected, axis=0)
            if not len(design) or (first_layer and decompose(design, tolerance, 'precision_rank') < design.shape[1] - 1):
                progress.update(index + 1, 'skipped: non-unique constraints')
                continue
            candidate = select_vector(decompose(design, rank_tol, 'precision_null'), old[columns])
            if candidate is None and first_layer:
                subset, _ = brute_force_max_row_subset_with_rank(design, len(columns))
                if subset is None:
                    choices = [x for x in find_good_rank_matrices(design, len(columns) + 1) if len(x)]
                    subset = max(choices, key=np.linalg.matrix_rank) if choices else None
                if subset is not None:
                    candidate = select_vector(decompose(subset, None, 'precision_null'), old[columns])
            if candidate is not None:
                # Align signs using the first shared nonzero coordinate to avoid division by zero.
                pivot = np.flatnonzero((old[columns] != 0) & (candidate[:-1] != 0))[0]
                candidate *= np.sign(old[columns][pivot] / candidate[pivot])
                weights[columns, index], biases[index] = candidate[:-1], candidate[-1]
                improved[index] = True
            progress.update(index + 1, f'improved={int(improved.sum())}')
    return weights, biases, improved


# Query-only signature refinement after source clustering.
# It uses redundant directions, held-out directions and agreement between steps;
# deep prefixes combine observable-coordinate constraints from multiple regions.
# Target parameters and error-detection tolerances are not modified.


def validate_refinement_options(direction_factor, max_witnesses, step_trials, tolerance, eps):
    for name, value, minimum in (('refinement_direction_factor', direction_factor, 2),
                                 ('refinement_max_witnesses', max_witnesses, 2),
                                 ('refinement_step_trials', step_trials, 2)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    if step_trials > 6:
        raise ValueError('refinement_step_trials must be <= 6')
    if isinstance(tolerance, bool) or not np.isfinite(tolerance) or not 0 < tolerance < 1:
        raise ValueError('refinement_tolerance must be finite and in (0, 1)')
    if not np.isfinite(eps) or eps <= 0 or eps > np.finfo(float).max / (10. ** (step_trials - 1)):
        raise ValueError('refinement steps must be positive and finite')


def _differences(query, point, directions, eps, output_index, advance):
    block_size = max(1, getattr(query, 'batch_size', 256) // 4)
    values = []
    inner = eps / 3
    for start in range(0, len(directions), block_size):
        block = directions[start:start + block_size]
        inputs = np.stack((point + (eps - inner) * block, point + eps * block,
                           point - (eps - inner) * block, point - eps * block), axis=1)
        output = query_numpy(query, inputs.reshape(-1, len(point)))[:, output_index].reshape(-1, 4)
        values.extend((output[:, 1] - output[:, 0] + output[:, 3] - output[:, 2]) / inner)
        advance(len(block))
    return np.asarray(values)


def _measure(query, point, directions, fit_count, eps, output_index, advance):
    raw = _differences(query, point, directions, eps, output_index, advance)
    # A strong fitting reference reduces orientation ambiguity near a zero kink.
    reference = int(np.argmax(np.abs(raw[:fit_count])))
    other = np.arange(len(directions)) != reference
    both = _differences(query, point, (directions[other] + directions[reference]) / 2,
                        eps, output_index, advance)
    positive = np.abs(np.abs(raw[reference] + raw[other]) / 2 - np.abs(both))
    negative = np.abs(np.abs(raw[reference] - raw[other]) / 2 - np.abs(both))
    values = raw.copy()
    values[other] *= np.where(negative < positive, -1, 1)
    amplitude = float(np.max(np.abs(raw)))
    error = float(np.max(np.minimum(positive, negative))) / max(amplitude, 1e-300)
    return values, error


def estimate_stable_direction(query, point, rng, *, eps=1e-5, output_index=0,
                              direction_factor=2, step_trials=3, tolerance=1e-4,
                              advance=lambda count: None):
    """Return a unit direction and diagnostics; None means insufficient evidence.

    At least two neighboring steps must agree in signed measurement amplitude,
    with low orientation and held-out fit errors. No target internals are used.
    """
    validate_refinement_options(direction_factor, 2, step_trials, tolerance, eps)
    point = np.asarray(point, dtype=np.float64)
    if point.ndim != 1 or not len(point) or not np.isfinite(point).all():
        raise ValueError('Expected a finite nonempty point vector')
    dimension = len(point)
    fit_count = direction_factor * (dimension + 2)
    validation_count = max(16, (dimension + 3) // 4)
    directions = np.sign(rng.normal(size=(fit_count + validation_count, dimension)))
    fitting, validation = directions[:fit_count], directions[fit_count:]
    measured, trials = [], []
    for trial in range(step_trials):
        step = eps * 10. ** trial
        y, orientation_error = _measure(query, point, directions, fit_count, step, output_index, advance)
        finite = bool(np.isfinite(y).all() and np.isfinite(orientation_error))
        measured.append(y if finite else np.zeros(len(directions)))
        trials.append(dict(eps=step, finite=finite,
                           orientation_error=orientation_error if finite else None))
    values = np.column_stack(measured)
    solutions, _, rank, singular = np.linalg.lstsq(fitting, values[:fit_count], rcond=1e-5)
    info = dict(fit_directions=fit_count, validation_directions=validation_count,
                rank=int(rank), condition_number=(float(singular[0] / singular[-1])
                                                  if singular[-1] > 0 else None), trials=trials)
    valid = []
    for index, trial in enumerate(trials):
        signal = float(np.linalg.norm(values[fit_count:, index]))
        ratio_norm = float(np.linalg.norm(solutions[:, index]))
        residual = float(np.linalg.norm(validation @ solutions[:, index] - values[fit_count:, index]))
        score = residual / max(signal, 1e-300)
        good = (rank == dimension and trial['finite'] and np.isfinite(signal) and np.isfinite(ratio_norm)
                and signal > 1e-12 and ratio_norm > 1e-12
                and np.isfinite(score) and trial['orientation_error'] <= tolerance and score <= tolerance)
        trial.update(validation_relative_residual=score if np.isfinite(score) else None,
                     signal_norm=signal if np.isfinite(signal) else None, usable=bool(good))
        valid.append(good)
    eligible = set()
    for index in range(1, step_trials):
        a, b = values[:, index - 1], values[:, index]
        sign = 1 if a @ b >= 0 else -1
        agreement = float(np.linalg.norm(a - sign * b) / max(np.linalg.norm(a), np.linalg.norm(b), 1e-300))
        trials[index]['previous_step_relative_difference'] = agreement if np.isfinite(agreement) else None
        if valid[index - 1] and valid[index] and agreement <= tolerance:
            eligible.update((index - 1, index))
    if not eligible:
        info['status'] = 'no_stable_step_pair'
        return None, info
    selected = min(eligible, key=lambda i: (trials[i]['validation_relative_residual'], -i))
    direction = solutions[:, selected]
    info.update(status='stable', selected_eps=trials[selected]['eps'],
                validation_relative_residual=trials[selected]['validation_relative_residual'])
    return direction / np.linalg.norm(direction), info


def _boundary_error(points, unit, bias):
    residual = np.abs(points @ unit + bias) / (1 + np.linalg.norm(points, axis=1))
    return np.quantile(residual, [.5, .9])


def refine_first_layer_signatures(query, weights, biases, groups, rng, *, eps=1e-5,
                                  output_index=0, direction_factor=2, max_witnesses=8,
                                  step_trials=3, tolerance=1e-4, diagnostics=None,
                                  progress_mode='auto', progress_interval=5.0, cache=None):
    """Fuse independently remeasured first-layer directions and robustly refit bias.

    The original column scale/sign is retained. Fewer than two stable witnesses,
    poor consensus, or worsened witness residuals leave the source column intact.
    diagnostics is updated incrementally, including when a query budget interrupts.
    """
    validate_refinement_options(direction_factor, max_witnesses, step_trials, tolerance, eps)
    weights = np.array(weights, dtype=np.float64, copy=True)
    biases = np.array(biases, dtype=np.float64, copy=True)
    if weights.ndim != 2 or biases.shape != (weights.shape[1],) or len(groups) != len(biases):
        raise ValueError('Weights, biases and witness groups must agree')
    dimension = weights.shape[0]
    diagnostics = {} if diagnostics is None else diagnostics
    diagnostics.update(method='stable_first_layer', refined=0, retained=0, neurons=[],
                       direction_factor=direction_factor, max_witnesses=max_witnesses,
                       step_trials=step_trials, tolerance=tolerance, eps=eps)
    planned = []
    for index, group in enumerate(groups):
        points = np.asarray(group, dtype=np.float64).reshape(-1, dimension)
        candidates = np.flatnonzero(np.isfinite(points).all(axis=1))
        candidates = candidates[np.argsort(np.linalg.norm(points[candidates], axis=1), kind='stable')]
        if (len(candidates) < 2 or not np.isfinite(weights[:, index]).all()
                or not np.isfinite(biases[index]) or np.linalg.norm(weights[:, index]) == 0
                or len(candidates) != len(points)):
            candidates = np.empty(0, dtype=int)
        planned.append((points, candidates[:max_witnesses]))
    direction_count = direction_factor * (dimension + 2) + max(16, (dimension + 3) // 4)
    units_per_point = step_trials * (2 * direction_count - 1)
    total = sum(len(indices) for _, indices in planned) * units_per_point
    completed = 0
    query_start = query.query_count
    scope = cache.scope(query) if cache is not None else None
    with Progress('layer=1 stable signature refinement', total, progress_interval, mode=progress_mode) as bar:
        try:
            for index, (points, indices) in enumerate(planned):
                entry = dict(candidate=index, witnesses=len(points), selected_witnesses=indices.tolist(),
                             status='retained_source', reason='insufficient_finite_witnesses', points=[])
                diagnostics['neurons'].append(entry)
                estimates = []
                norm = np.linalg.norm(weights[:, index])
                reference = weights[:, index] / norm if norm else weights[:, index]
                for witness in indices:
                    point_info = dict(witness=int(witness), status='measuring')
                    entry['points'].append(point_info)
                    def advance(count):
                        nonlocal completed
                        completed += count
                        bar.update(completed, f'candidate={index + 1}/{len(biases)} witness={witness} '
                                   f'refined={diagnostics["refined"]} retained={diagnostics["retained"]}')
                    key = (cache.key(scope, 'stable_first', points[witness], eps, output_index,
                                     direction_factor, step_trials, tolerance)
                           if cache is not None else None)
                    reused = cache.get(key) if cache is not None else None
                    if reused is None:
                        estimate, details = estimate_stable_direction(query, points[witness], rng, eps=eps,
                            output_index=output_index, direction_factor=direction_factor, step_trials=step_trials,
                            tolerance=tolerance, advance=advance)
                        if cache is not None and estimate is not None:
                            cache.put(key, (estimate, details))
                    else:
                        estimate, details = reused
                        rng.normal(size=(direction_count, dimension))
                        advance(units_per_point)
                        details['cached'] = True
                    point_info.update(details)
                    if estimate is not None:
                        estimates.append(estimate * (1 if estimate @ reference >= 0 else -1))
                entry['stable_witnesses'] = len(estimates)
                if len(estimates) >= 2:
                    estimates = np.asarray(estimates)
                    distances = np.linalg.norm(estimates[:, None] - estimates[None, :], axis=2)
                    inliers = distances[np.argmax((distances <= tolerance).sum(axis=1))] <= tolerance
                    entry['consensus_witnesses'] = int(inliers.sum())
                    entry['reason'] = 'insufficient_direction_consensus'
                    if inliers.sum() >= 2:
                        fused = estimates[inliers].mean(axis=0)
                        fused /= np.linalg.norm(fused)
                        bias = float(np.median(-(points @ fused)))
                        old_error = _boundary_error(points, reference, biases[index] / norm)
                        new_error = _boundary_error(points, fused, bias)
                        entry.update(old_boundary_error=old_error.tolist(), new_boundary_error=new_error.tolist())
                        entry['reason'] = 'witness_residual_worsened'
                        if np.all(new_error <= old_error + 1e-12):
                            weights[:, index], biases[index] = fused * norm, bias * norm
                            entry.update(status='refined', reason='stable_consensus_and_witness_validation')
                elif len(indices):
                    entry['reason'] = 'insufficient_stable_witnesses'
                diagnostics['refined' if entry['status'] == 'refined' else 'retained'] += 1
        finally:
            diagnostics['queries'] = query.query_count - query_start
    log_progress(f'First-layer refinement: refined={diagnostics["refined"]} retained={diagnostics["retained"]} '
                 f'queries={diagnostics["queries"]}')
    return weights, biases


def _estimate_active_direction(query, prefix, point, rng, *, eps, output_index,
                               direction_factor, step_trials, tolerance):
    """Estimate only observable coordinates; never fill inactive coordinates with zero."""
    hidden = prefix_values(prefix, point).reshape(-1)
    active = np.flatnonzero(hidden > 0)
    info = dict(active=active.tolist(), status='no_stable_step_pair', trials=[])
    if len(active) < 2 or not np.isfinite(hidden).all():
        return None, info
    fit = direction_factor * (len(active) + 2)
    validation = max(16, len(active) // 2)
    directions = np.sign(rng.normal(size=(fit + validation, len(point))))
    with torch.no_grad():
        matrix = prefix.forward_at(prefix_tensor(prefix, point), prefix_tensor(prefix, directions)).cpu().numpy()[:, active]
    if not np.isfinite(matrix).all():
        return None, info
    measured = []
    for trial in range(step_trials):
        step = eps * 10. ** trial
        y, orientation = _measure(query, point, directions, fit, step, output_index, lambda _: None)
        if not np.isfinite(y).all() or not np.isfinite(orientation):
            info['status'] = 'nonfinite_measurement'
            return None, info
        measured.append(y)
        info['trials'].append(dict(eps=step, orientation_error=orientation))
    values = np.column_stack(measured)
    solutions, _, rank, singular = np.linalg.lstsq(matrix[:fit], values[:fit], rcond=1e-5)
    info.update(rank=int(rank), condition_number=float(singular[0] / singular[-1]) if singular[-1] > 0 else None)
    valid = []
    for index, trial in enumerate(info['trials']):
        signal = np.linalg.norm(values[fit:, index])
        score = np.linalg.norm(matrix[fit:] @ solutions[:, index] - values[fit:, index]) / max(signal, 1e-300)
        usable = bool(rank == len(active) and np.isfinite(score) and signal > 1e-12
                      and np.linalg.norm(solutions[:, index]) > 1e-12
                      and trial['orientation_error'] <= tolerance and score <= tolerance)
        trial.update(validation_relative_residual=float(score) if np.isfinite(score) else None, usable=usable)
        valid.append(usable)
    eligible = set()
    for index in range(1, step_trials):
        a, b = values[:, index - 1], values[:, index]
        sign = 1 if a @ b >= 0 else -1
        delta = np.linalg.norm(a - sign * b) / max(np.linalg.norm(a), np.linalg.norm(b), 1e-300)
        info['trials'][index]['previous_step_relative_difference'] = float(delta) if np.isfinite(delta) else None
        if valid[index - 1] and valid[index] and delta <= tolerance:
            eligible.update((index - 1, index))
    if not eligible:
        return None, info
    selected = min(eligible, key=lambda i: info['trials'][i]['validation_relative_residual'])
    unit = solutions[:, selected] / np.linalg.norm(solutions[:, selected])
    info.update(status='stable', selected_eps=info['trials'][selected]['eps'])
    return (active, unit), info


def refine_deep_layer_signatures(query, prefix, weights, biases, groups, rng, *, eps=1e-5,
                                 output_index=0, direction_factor=2, step_trials=3,
                                 tolerance=1e-4, diagnostics=None,
                                 progress_mode='auto', progress_interval=5.0, cache=None):
    """Fuse local active-coordinate constraints across all witnesses of each cluster.

    Require observable coordinates, a unique direction and non-worsening boundary
    residuals. Original column scale/sign is retained; insufficient evidence falls
    back to the source parameters. No target internals or true width are used.
    """
    validate_refinement_options(direction_factor, 2, step_trials, tolerance, eps)
    weights, biases = np.array(weights, dtype=float, copy=True), np.array(biases, dtype=float, copy=True)
    if (not len(prefix.fcs) or weights.ndim != 2 or weights.shape[0] != prefix.structure[-1]
            or biases.shape != (weights.shape[1],) or len(groups) != len(biases)):
        raise ValueError('Expected a nonempty prefix and matching signatures/groups')
    diagnostics = {} if diagnostics is None else diagnostics
    diagnostics.update(method='stable_active_coordinates', neurons=[], refined=0, retained=0,
                       eps=eps, direction_factor=direction_factor, step_trials=step_trials, tolerance=tolerance)
    prepared = []
    for index, group in enumerate(groups):
        points = np.asarray(group, dtype=float).reshape(-1, prefix.structure[0])
        norm = np.linalg.norm(weights[:, index])
        valid = (len(points) >= 2 and np.isfinite(points).all() and np.isfinite(norm) and norm > 0
                 and np.isfinite(biases[index]))
        prepared.append(points if valid else points[:0])
    start = query.query_count
    scope = cache.scope(query, prefix) if cache is not None else None
    completed = 0
    with Progress(f'layer={len(prefix.fcs) + 1} stable active refinement', sum(map(len, prepared)),
                  progress_interval, mode=progress_mode) as bar:
        try:
            for index, points in enumerate(prepared):
                entry = dict(candidate=index, witnesses=len(groups[index]), points=[], status='retained_source',
                             reason='insufficient_stable_evidence')
                diagnostics['neurons'].append(entry)
                measured = []
                for witness, point in enumerate(points):
                    detail = dict(witness=witness, status='measuring')
                    entry['points'].append(detail)
                    key = (cache.key(scope, 'stable_deep', point, eps, output_index,
                                     direction_factor, step_trials, tolerance)
                           if cache is not None else None)
                    reused = cache.get(key) if cache is not None else None
                    if reused is None:
                        value, information = _estimate_active_direction(query, prefix, point, rng, eps=eps,
                            output_index=output_index, direction_factor=direction_factor,
                            step_trials=step_trials, tolerance=tolerance)
                        if cache is not None and value is not None:
                            cache.put(key, (value, information))
                    else:
                        value, information = reused
                        active_count = len(value[0])
                        rng.normal(size=(direction_factor * (active_count + 2) +
                                         max(16, active_count // 2), len(point)))
                        information['cached'] = True
                    detail.update(information)
                    if value is not None:
                        measured.append(value)
                    completed += 1
                    bar.update(completed, f'candidate={index + 1}/{len(groups)} stable={len(measured)} '
                               f'refined={diagnostics["refined"]} queries={query.query_count - start}')
                entry['stable_witnesses'] = len(measured)
                if len(measured) >= 2:
                    dimension = weights.shape[0]
                    gram = np.zeros((dimension, dimension))
                    coverage = np.zeros(dimension, dtype=int)
                    for active, unit in measured:
                        gram[np.ix_(active, active)] += np.eye(len(active)) - np.outer(unit, unit)
                        coverage[active] += 1
                    eigenvalues, vectors = np.linalg.eigh(gram)
                    unit = vectors[:, 0]
                    unit *= 1 if unit @ weights[:, index] >= 0 else -1
                    residuals = [np.linalg.norm(unit[a] - u * (unit[a] @ u)) /
                                 max(np.linalg.norm(unit[a]), 1e-300) for a, u in measured]
                    entry.update(coverage=coverage.tolist(), eigenvalues=eigenvalues[:3].tolist(),
                                 consensus_residuals=[float(v) for v in residuals])
                    entry['reason'] = 'unobservable_or_ambiguous_direction'
                    if (dimension >= 2 and coverage.min() >= 2 and eigenvalues[1] > tolerance
                            and np.isfinite(residuals).all() and max(residuals) <= tolerance):
                        hidden = prefix_values(prefix, points)
                        bias = float(np.median(-(hidden @ unit)))
                        norm = np.linalg.norm(weights[:, index])
                        old_error = _boundary_error(hidden, weights[:, index] / norm, biases[index] / norm)
                        new_error = _boundary_error(hidden, unit, bias)
                        entry.update(old_boundary_error=old_error.tolist(), new_boundary_error=new_error.tolist(),
                                     reason='witness_residual_worsened')
                        if np.all(new_error <= old_error + 1e-12):
                            weights[:, index], biases[index] = unit * norm, bias * norm
                            entry.update(status='refined', reason='stable_active_constraints_and_witness_validation')
                diagnostics['refined' if entry['status'] == 'refined' else 'retained'] += 1
        finally:
            diagnostics['queries'] = query.query_count - start
    log_progress(f'Active-coordinate refinement: refined={diagnostics["refined"]} '
                 f'retained={diagnostics["retained"]} queries={diagnostics["queries"]}')
    return weights, biases
