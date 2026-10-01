"""Recover hidden signatures and output layers, and handle stalled candidates."""
from extraction.utils import query_numpy, RecoveryCache
import copy
from scipy.optimize import linear_sum_assignment
from extraction.sign_recovery import recover_signs
from dataclasses import dataclass
from argparse import Namespace
import numpy as np
import torch
from scipy.linalg import null_space
from extraction.utils import prefix_tensor, prefix_values, cached_prefix_values
from extraction.clustering import basic_cluster
from extraction.clustering import source_graph
from extraction.clustering import ratio_normalize
from extraction.clustering import merge_bad_critical_points_into_components, merge_bad_critical_points_with_each_other
from utils.progress import Progress, log_progress


class InsufficientEvidence(RuntimeError):
    """Signal that the available evidence cannot establish complete recovery."""


@dataclass
class LayerSignatures:
    weights: np.ndarray
    biases: np.ndarray
    groups: list
    filtered: int
    failed: int
    candidates: int
    rank_deficient: int = 0
    rank_unmerged: int = 0

    @property
    def complete(self):
        return bool(len(self.biases)) and np.all(np.isfinite(self.weights)) and np.all(np.isfinite(self.biases))


def _directional_differences(query, point, directions, eps, output_index):
    """Query original and combined directions in order, prefetching at most one block."""
    block_size = max(1, getattr(query, 'batch_size', 256) // 4)
    inner = eps / 3
    total = 2 * len(directions) - 1
    for start in range(0, total, block_size):
        block = []
        for index in range(start, min(start + block_size, total)):
            direction = directions[(index + 1) // 2]
            if index and index % 2 == 0:
                direction = (direction + directions[0]) / 2
            block.append(direction)
        block = np.asarray(block)
        points = np.stack((point + (eps - inner) * block, point + eps * block,
                           point - (eps - inner) * block, point - eps * block), axis=1)
        outputs = query_numpy(query, points.reshape(-1, directions.shape[1]))[:, output_index].reshape(-1, 4)
        yield from (outputs[:, 1] - outputs[:, 0] + outputs[:, 3] - outputs[:, 2]) / inner


def _solve_ratio(matrix, values, active, *, full_matrix_rank=False):
    """Reuse least-squares singular values with matrix_rank's default threshold.

    Keep the full matrix in the solve.
    """
    ratio, _, _, singular = np.linalg.lstsq(matrix, values, rcond=1e-5)
    active_count = int(np.count_nonzero(active))
    if full_matrix_rank:
        threshold = singular[0] * max(matrix.shape) * np.finfo(singular.dtype).eps
        rank = np.count_nonzero(singular > threshold)
    elif np.all(active) or not np.any(matrix[:, ~active]):
        # Removing exact zero columns preserves nonzero singular values; use the active shape for the threshold.
        threshold = singular[0] * max(len(matrix), active_count) * np.finfo(singular.dtype).eps
        rank = np.count_nonzero(singular[:min(len(matrix), active_count)] > threshold)
    else:
        # Small inactive columns may not be zero, so the full matrix spectrum cannot be reused.
        rank = np.linalg.matrix_rank(matrix[:, active])
    return ratio, rank < active_count


def estimate_ratio(query, prefix, point, rng, *, output_index=0, eps=1e-5,
                   return_constraints=False, return_details=False):
    """Estimate ratios with N+2 random sign directions; mark inactive coordinates as NaN.

    Use a 1e-2 sign-consistency threshold and lstsq rcond=1e-5.
    Path-dependent real_attack column normalization is not applied.
    """
    # An empty prefix is the identity; zero input coordinates remain controllable.
    with torch.no_grad():
        preactivation = prefix.forward(prefix_tensor(prefix, point), with_relu=False).cpu().numpy()
    count = int(np.count_nonzero(preactivation)) + 2
    directions = np.sign(rng.normal(size=(count, prefix.structure[0])))
    differences = iter(_directional_differences(query, point, directions, eps, output_index))
    values = []
    for index, direction in enumerate(directions):
        value = next(differences)
        if index:
            both = next(differences)
            positive = abs(abs(values[0] + value) / 2 - abs(both))
            negative = abs(abs(values[0] - value) / 2 - abs(both))
            if min(positive, negative) > 1e-2:
                raise InsufficientEvidence('Inconsistent directional differences')
            if negative < positive:
                value *= -1
        values.append(value)
    with torch.no_grad():
        matrix = prefix.forward_at(prefix_tensor(prefix, point),
                                   prefix_tensor(prefix, directions)).cpu().numpy()
    active = ~(np.mean(np.abs(matrix) < 1e-8, axis=0) > 0.5)
    if not len(prefix.fcs):
        active = np.ones(matrix.shape[1], dtype=bool)
    if not np.any(active):
        raise InsufficientEvidence('No active prefix directions')
    values = np.asarray(values)
    ratio, deficient = _solve_ratio(matrix, values, active, full_matrix_rank=return_details)
    if deficient and not (return_constraints or return_details):
        raise InsufficientEvidence('Rank deficient local prefix directions')
    ratio[~active] = np.nan
    if return_details:
        return ratio, matrix, deficient if len(prefix.fcs) else False
    scale = np.nanmax(np.abs(ratio))
    if not np.isfinite(scale) or scale < 1e-12:
        raise InsufficientEvidence('No observable output kink')
    ratio /= scale
    if return_constraints:
        hidden = prefix_values(prefix, point).reshape(-1)
        # Homogeneous constraints support diagnostics only, not extraction clustering.
        norm2 = float(values @ values)
        if norm2 < 1e-24:
            raise InsufficientEvidence('No observable output kink')
        constraints = matrix - np.outer(values, values @ matrix) / norm2
        constraints = np.column_stack((constraints, np.zeros(len(constraints))))
        constraints = np.vstack((constraints, np.append(hidden, 1)))
        lengths = np.linalg.norm(constraints, axis=1)
        constraints = constraints[lengths > 1e-12] / lengths[lengths > 1e-12, None]
    return (ratio, constraints, deficient) if return_constraints else ratio


def _signature_evidence(query, prefix, point, rng, eps, output_index, cache, scope):
    """Reuse each point's first successful measurement and reconstruct its direction matrix.

    Avoid caching the large per-point local matrix.
    """
    generator_kind = type(rng.bit_generator).__name__ if isinstance(rng, np.random.Generator) else None
    key = cache.key(scope, 'signature', point, eps, output_index, generator_kind) if cache is not None else None
    cached = cache.get(key) if cache is not None and isinstance(rng, np.random.Generator) else None
    if cached is not None:
        ratio, count, state, kernel, deficient = cached
        # Preserve RNG draws and reconstruct H from the first successful measurement's directions.
        rng.normal(size=(count, prefix.structure[0]))
        replay = np.random.Generator(type(rng.bit_generator)())
        replay.bit_generator.state = state
        directions = np.sign(replay.normal(size=(count, prefix.structure[0])))
        with torch.no_grad():
            matrix = prefix.forward_at(prefix_tensor(prefix, point),
                                       prefix_tensor(prefix, directions)).cpu().numpy()
        return ratio, matrix, deficient, kernel
    state = copy.deepcopy(rng.bit_generator.state) if isinstance(rng, np.random.Generator) else None
    ratio, matrix, deficient = estimate_ratio(query, prefix, point, rng, output_index=output_index,
                                              eps=eps, return_details=True)
    kernel = []
    if deficient:
        active = ~(np.mean(np.abs(matrix) < 1e-8, axis=0) > 0.5)
        reduced = null_space(matrix[:, active])
        kernel = np.zeros((matrix.shape[1], reduced.shape[1]))
        kernel[active] = reduced
    if cache is not None and state is not None:
        cache.put(key, (ratio, len(matrix), state, kernel, deficient))
    return ratio, matrix, deficient, kernel


def recover_layer_weights(query, prefix, points, width, rng, *, output_index=0,
                          eps=1e-5, tolerance=1e-4, min_witnesses=3,
                          bad_pair_budget=100,
                          cluster_method='source',
                          first_layer_clustering='geometry', first_layer_direction_tolerance=1e-4,
                          first_layer_bias_tolerance=1e-5, clustering_diagnostics=None,
                          progress_mode='auto', progress_interval=5.0, cache=None):
    """Filter, estimate ratios, graph-cluster, then recover biases.

    Select sampled-axis or all-axis graph clustering, followed by optional
    first-layer direction and bias splitting. tolerance and bad_pair_budget
    are accepted for compatibility; kernel merging uses fixed thresholds and
    examines all eligible pairs.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, prefix.structure[0])
    if first_layer_clustering not in ('geometry', 'source'):
        raise ValueError('first_layer_clustering must be geometry or source')
    for value in (first_layer_direction_tolerance, first_layer_bias_tolerance):
        if not np.isfinite(value) or value <= 0:
            raise ValueError('First-layer geometry tolerances must be finite and positive')
    if clustering_diagnostics is not None:
        clustering_diagnostics.update(mode=first_layer_clustering, applied=False)
    input_indices = np.arange(len(points))
    components, rows, bad = [], [], []
    ratios, ratio_points = [], []
    criticals, special_solutions, kernels, good_ids, bad_ids = [], [], [], [], []
    filtered = failed = 0
    scope = cache.scope(query, prefix) if cache is not None else None
    label = f'layer={len(prefix.fcs) + 1} signatures'
    with Progress(label, len(points), progress_interval, mode=progress_mode) as progress:
        for index, point in enumerate(points):
            if prefix.prefix_boundary_mask(point).item():
                filtered += 1
            else:
                try:
                    ratio, matrix, deficient, kernel = _signature_evidence(
                        query, prefix, point, rng, eps, output_index, cache, scope)
                    cp_id = len(criticals)
                    criticals.append(Namespace(point=point, h_matrix=matrix, ratio=ratio,
                                               input_index=int(input_indices[index])))
                    special_solutions.append(ratio)
                    if deficient:
                        kernels.append(kernel)
                        bad_ids.append(cp_id)
                    else:
                        kernels.append([])
                        good_ids.append(cp_id)
                    if deficient:
                        bad.append(point)
                        progress.update(index + 1, f'ratios={len(ratios)} rank_deficient={len(bad)} '
                                        f'filtered={filtered} failed={failed}')
                        continue
                    ratios.append(ratio)
                    ratio_points.append(point)
                except (InsufficientEvidence, np.linalg.LinAlgError):
                    failed += 1
            progress.update(index + 1, f'ratios={len(ratios)} filtered={filtered} failed={failed}')
    matrix = np.asarray(ratios, dtype=np.float64).reshape(-1, prefix.structure[-1])
    block_size = 128
    blocks = (len(matrix) + block_size - 1) // block_size
    total = matrix.shape[1] * blocks * (blocks + 1) // 2
    if cluster_method == 'source':
        clustered = source_graph(matrix, rng, first_layer=not len(prefix.fcs),
                                 progress_mode=progress_mode, progress_interval=progress_interval, cache=cache)
    elif cluster_method == 'basic':
        with Progress(f'{label} graph clustering', total, progress_interval,
                      mode=progress_mode) as progress:
            clustered = basic_cluster(matrix, block_size=block_size, progress=progress)
    else:
        raise ValueError('cluster_method must be source or basic')
    if first_layer_clustering == 'geometry' and not len(prefix.fcs):
        from extraction.clustering import split_first_layer_groups
        if clustering_diagnostics is not None:
            clustering_diagnostics.update(applied=True,
                ratio_input_indices=[criticals[i].input_index for i in good_ids])
        clustered = split_first_layer_groups(matrix, ratio_points, clustered,
            direction_tolerance=first_layer_direction_tolerance,
            bias_tolerance=first_layer_bias_tolerance, diagnostics=clustering_diagnostics,
            progress_mode=progress_mode, progress_interval=progress_interval)
    components = [[good_ids[i] for i in indices] for indices in clustered]
    for indices in clustered:
        direction = (matrix[indices[0]].copy() if not len(prefix.fcs) or len(indices) == 1
                     else ratio_normalize(matrix[indices], verbose=False, cache=cache)[0])
        if not len(prefix.fcs):
            direction /= np.max(np.abs(direction))
        rows.append(direction)
    rows = np.asarray(rows).reshape(-1, prefix.structure[-1])

    def progress_iter(values, desc):
        with Progress(f'{label} {desc}', len(values), progress_interval,
                      mode=progress_mode) as progress:
            for index, value in enumerate(values):
                yield value
                progress.update(index + 1)

    args = Namespace(dataset='cifar10', real_attack=0, layerID=len(prefix.fcs) + 1,
                     debug=False, progress_iter=progress_iter)
    if bad_ids:
        components, rows, merged = merge_bad_critical_points_into_components(
            args, None, criticals, special_solutions, kernels, bad_ids, components, rows)
        remaining_bad = [cp for i, cp in enumerate(bad_ids) if i not in merged]
        assigned = {cp for group in components if len(group) > 1 for cp in group}
        remaining_good = [cp for cp in good_ids if cp not in assigned]
        # Singletons are passed as unmerged good points, not duplicate components.
        kept = [i for i, group in enumerate(components) if len(group) > 1]
        singleton_components = [components[i] for i in range(len(components)) if i not in kept]
        singleton_rows = [rows[i] for i in range(len(components)) if i not in kept]
        components, rows, _ = merge_bad_critical_points_with_each_other(
            args, None, criticals, special_solutions, kernels, remaining_bad, remaining_good,
            [components[i] for i in kept], np.asarray([rows[i] for i in kept]).reshape(-1, prefix.structure[-1]))
        assigned = {cp for group in components for cp in group}
        rows = list(rows)
        for group, row in zip(singleton_components, singleton_rows):
            if group[0] not in assigned:
                components.append(group)
                rows.append(row)
    assigned = {cp for group in components for cp in group}
    unmerged = sum(cp not in assigned for cp in bad_ids)
    components = [[criticals[i].point for i in group] for group in components]
    ordered = sorted((i for i in range(len(rows)) if len(components[i]) >= min_witnesses),
                     key=lambda i: (-len(components[i]), np.isnan(rows[i]).sum()))
    candidates = len(ordered)
    if width is None:
        # Keep every supported component; do not pad or truncate to a ground-truth width.
        width = candidates
    weights = np.full((prefix.structure[-1], width), np.nan)
    biases = np.full(width, np.nan)
    groups = [np.empty((0, prefix.structure[0])) for _ in range(width)]
    with Progress(f'{label} biases', min(len(ordered), width), progress_interval,
                  mode=progress_mode) as progress:
        for column, index in enumerate(ordered[:width]):
            weights[:, column] = rows[index]
            groups[column] = np.asarray(components[index])
            if np.all(np.isfinite(weights[:, column])):
                # Use 1e-3 bias voting without replacing merged weights with a single local ratio.
                values = -(cached_prefix_values(prefix, groups[column], cache) @ weights[:, column])
                votes = []
                for value in values:
                    for bucket in votes:
                        if abs(value - bucket[0]) < 1e-3:
                            bucket[1] += 1
                            break
                    else:
                        votes.append([value, 1])
                biases[column] = max(votes, key=lambda item: item[1])[0]
            progress.update(column + 1, f'recovered={int(np.isfinite(biases).sum())}/{width}')
    return LayerSignatures(weights, biases, groups, filtered, failed, candidates, len(bad), unmerged)


def recover_layer_weights_unknown(query, prefix, points, rng, **kwargs):
    """Infer candidate width from supported components.

    complete means all candidate parameters are finite, not that the true layer is
    fully recovered. Signs, remaining-point coverage, and detection still require validation.
    """
    return recover_layer_weights(query, prefix, points, None, rng, **kwargs)


def recover_output_layer(query, prefix, points, *, batch_size=256, tolerance=1e-5,
                         progress_mode='auto', progress_interval=5.0):
    matrices, outputs = [], []
    with Progress('output linear system', len(points), progress_interval, mode=progress_mode) as progress:
        for start in range(0, len(points), batch_size):
            batch = points[start:start + batch_size]
            hidden = prefix_values(prefix, batch)
            matrices.append(np.column_stack((hidden, np.ones(len(hidden)))))
            outputs.append(query_numpy(query, batch))
            progress.update(start + len(batch))
    if not matrices:
        raise InsufficientEvidence('No samples for output recovery')
    design, observed = np.concatenate(matrices), np.concatenate(outputs)
    with Progress('solve output linear system', 1, progress_interval, mode=progress_mode) as progress:
        solution, _, rank, singular = np.linalg.lstsq(design, observed, rcond=None)
        residual = float(np.max(np.abs(design @ solution - observed)))
        if rank < design.shape[1]:
            raise InsufficientEvidence(f'Output design rank {rank}/{design.shape[1]}: parameters not identifiable')
        if residual > tolerance:
            raise InsufficientEvidence(f'Output residual {residual:g} exceeds {tolerance:g}')
        progress.update(1, f'rank={rank}/{design.shape[1]} residual={residual:.3g}')
    return solution[:-1], solution[-1], dict(rank=int(rank), columns=design.shape[1],
                                            max_abs_residual=residual,
                                            singular_values=singular.tolist())


# Query-only cross-layer hypotheses for persistent contributions.
# A stalled candidate is not accepted as a complete layer. Its observable ReLU
# coordinates and linear input are used together to recover the following layer.


def recover_cross_layer_weights(query, prefix, points, rng, config):
    """Intersect homogeneous local-normal and boundary constraints across regions.

    Expanded coordinates are necessarily locally dependent: ordinary-neuron
    ratios cannot identify C in one activation region. A candidate is emitted
    only when compatible constraints leave a one-dimensional null space.
    Pair seeds and greedy region extension are bounded; failure retains evidence.
    """
    constraints, witnesses, jacobians = [], [], []
    failed = filtered = 0
    dim = prefix.structure[-1] + 1
    tolerance = config.persistent_constraint_tolerance
    # Retain the entire pool in the caller; only bound this expensive hypothesis fit.
    if len(points) > config.persistent_max_points:
        points = points[rng.choice(len(points), config.persistent_max_points, replace=False)]
    with Progress('cross-layer local constraints', len(points), config.progress_interval,
                  mode=config.progress_mode) as bar:
        for index, point in enumerate(points):
            if prefix.prefix_boundary_mask(point).item():
                filtered += 1
            else:
                try:
                    _, matrix, _ = estimate_ratio(query, prefix, point, rng,
                        output_index=config.output_index, eps=config.ratio_eps, return_constraints=True)
                    if np.isfinite(matrix).all():
                        constraints.append(matrix)
                        witnesses.append(point)
                        jacobians.append(prefix.forward_at(
                            torch.as_tensor(point, device=prefix.device, dtype=prefix.dtype),
                            torch.as_tensor(rng.normal(size=(4, prefix.structure[0])),
                                            device=prefix.device, dtype=prefix.dtype)
                        ).cpu().numpy())
                    else:
                        failed += 1
                except (InsufficientEvidence, np.linalg.LinAlgError):
                    failed += 1
            bar.update(index + 1, f'constraints={len(constraints)} failed={failed} filtered={filtered}')
    count = len(constraints)
    vectors, groups = [], []
    # A random bounded subset of unordered pairs avoids quadratic materialization.
    offsets = np.arange(count, dtype=np.int64)
    offsets = offsets * (2 * count - offsets - 1) // 2
    total = count * (count - 1) // 2
    seeds = rng.choice(total, size=min(total, config.persistent_pair_budget), replace=False)

    def kernel(matrices):
        matrix = np.concatenate(matrices)
        _, singular, right = np.linalg.svd(matrix, full_matrices=len(matrix) < dim)
        rank = int(np.count_nonzero(singular > tolerance * max(1., singular[0])))
        return right[rank:].T

    with Progress('cross-layer region intersections', len(seeds), config.progress_interval,
                  mode=config.progress_mode) as bar:
        for step, seed in enumerate(seeds):
            first = int(np.searchsorted(offsets, seed, side='right') - 1)
            second = first + 1 + int(seed - offsets[first])
            selected = [first, second]
            basis = kernel([constraints[i] for i in selected])
            if basis.shape[1] > 1:
                for other in rng.permutation(count):
                    if other in selected:
                        continue
                    trial = kernel([constraints[i] for i in selected] + [constraints[other]])
                    if trial.shape[1]:
                        basis = trial
                        selected.append(int(other))
                    if basis.shape[1] == 1:
                        break
            if basis.shape[1] == 1:
                vector = basis[:, 0]
                norm = np.linalg.norm(vector[:-1])
                if norm > tolerance:
                    vector = vector / norm
                    support = [i for i, matrix in enumerate(constraints)
                               if np.max(np.abs(matrix @ vector)) <= tolerance * max(1., np.linalg.norm(vector))
                               and np.linalg.norm(jacobians[i] @ vector[:-1]) > tolerance]
                    duplicate = any(min(np.linalg.norm(vector - old), np.linalg.norm(vector + old))
                                    <= config.persistent_match_tolerance for old in vectors)
                    if (len(support) >= config.min_witnesses and not duplicate
                            and kernel([constraints[i] for i in support]).shape[1] == 1):
                        vectors.append(vector)
                        groups.append(np.asarray([witnesses[i] for i in support]))
            bar.update(step + 1, f'candidates={len(vectors)} witnesses={count}')
    parameters = np.asarray(vectors).reshape(-1, dim)
    return LayerSignatures(parameters[:, :-1].T, parameters[:, -1], groups, filtered, failed,
                           len(vectors), count, count - len({tuple(p) for group in groups for p in group}))


class PersistentStallTracker:
    """Require fresh points, matched signed candidates and informative rejection.

    Candidate identity is matched up to positive scale and permutation, not by
    cluster index or count. This is a heuristic, never a persistence proof.
    """
    def __init__(self, tolerance):
        self.tolerance = tolerance
        self.previous = None
        self.pool_size = 0
        self.rounds = 0

    def observe(self, proposal, detection, pool_size):
        informative = (proposal is not None and detection is not None
                       and not detection['passed'] and detection.get('sufficient_tests', False)
                       and detection.get('inconsistent_rate', 0) is not None
                       and detection['inconsistent_rate'] > detection['thresholds']['max_inconsistent_rate']
                       and detection['skipped_rate'] <= detection['thresholds']['max_skipped_rate'])
        if not informative:
            self.previous, self.rounds, self.pool_size = None, 0, pool_size
            return 0
        weight = proposal.fcs[-1].weight.detach().cpu().numpy()
        bias = proposal.fcs[-1].bias.detach().cpu().numpy()
        norms = np.linalg.norm(weight, axis=1)
        if not len(norms) or np.any(norms == 0) or not np.isfinite(norms).all():
            self.previous, self.rounds = None, 0
            return 0
        current = np.column_stack((weight, bias)) / norms[:, None]
        same = False
        if self.previous is not None and self.previous.shape == current.shape:
            distance = np.linalg.norm(self.previous[:, None] - current[None, :], axis=2)
            rows, columns = linear_sum_assignment(distance)
            same = bool(np.all(distance[rows, columns] <= self.tolerance))
        self.rounds = self.rounds + 1 if same and pool_size > self.pool_size else 0
        self.previous, self.pool_size = current.copy(), pool_size
        return self.rounds


def recover_cross_layer(query, ordinary, state, quarantine, rng, detection_rng,
                        validation_rng, config, directory):
    """Return only a jointly validated hypothesis; state retains newly found points.

    The next weight matrix has blocks [A_R; C] and an absorbed bias. No attempt
    is made to factor C into individual persistent neurons or infer their count.
    """
    from extraction.pipeline import save_json, collect_points_unknown, _merge_pool, _active_remaining, remaining_points, detect_unknown_prefix, _finish_candidate, PointQuarantine, termination_coverage

    from extraction.precision import improve_precision

    directory.mkdir(parents=True)
    layer = len(ordinary.fcs)
    width = ordinary.structure[-1]
    extended = copy.deepcopy(ordinary).extend_last_layer_with_linear_input()
    metadata = dict(layer=layer, ordinary_width=width, linear_input_width=ordinary.structure[-2],
                    representation_width=extended.structure[-1], persistent_count=None,
                    status='pending', attempts=[], interpretation='possible persistent contribution')
    progress = dict(progress_mode=config.progress_mode, progress_interval=config.progress_interval)
    cross_cache = RecoveryCache()
    torch.save(extended.state_dict(), directory / 'extended_candidate.pth')
    save_json(directory / 'boundary_filter.json', extended.boundary_filter_profile())
    try:
        for attempt in range(config.persistent_max_rounds):
            trial = directory / f'round_{attempt + 1}'
            trial.mkdir()
            item = dict(round=attempt + 1, accepted=False)
            metadata['attempts'].append(item)
            log_progress(f'Persistent hypothesis layer={layer}: cross-layer round={attempt + 1}; '
                         'original prefix remains uncommitted')
            if attempt:
                extra = collect_points_unknown(query, rng, config, f'cross-layer {layer + 1} resample')
                state['pool'] = _merge_pool(state['pool'], extra)
            remaining, _, unexplained, _ = _active_remaining(extended, state['pool'], quarantine, config,
                                                   'cross-layer recovery coverage')
            np.save(trial / 'recovery_points.npy', remaining)
            item['termination_coverage'] = termination_coverage(int(unexplained.sum()), len(state['pool']), config)
            if item['termination_coverage']['passed']:
                # Here i+1 may be the affine output, checked against every logit.
                candidate, state['pool'], finish = _finish_candidate(query, extended, state['pool'],
                    rng, validation_rng, config, trial, quarantine=quarantine, detection_rng=detection_rng,
                    retained_state=state)
                item['finish'] = finish
                if candidate is not None:
                    item['accepted'] = True
                    metadata.update(status='accepted', output_layer=True, accepted_through=layer,
                                    finish=finish)
                    return candidate, metadata, []
                continue
            if layer >= config.max_hidden_layers:
                item['reason'] = 'next_hidden_layer_exceeds_safety_budget'
                break
            result = recover_cross_layer_weights(query, extended, remaining, rng, config)
            np.savez(trial / 'source_signatures.npz', weights=result.weights, biases=result.biases)
            np.savez(trial / 'groups.npz', **{f'neuron_{i}': group for i, group in enumerate(result.groups)})
            item.update(candidates=len(result.biases), rank_unmerged=result.rank_unmerged,
                        reason='incomplete_cross_layer_parameters')
            # Do not fill missing constraints with zeros or accept a partial second layer.
            if not result.complete:
                continue
            if config.improve_precision:
                result.weights, result.biases, _ = improve_precision(extended, result.weights,
                    result.biases, result.groups, tolerance=config.precision_rank_tolerance,
                    rng=rng, cache=cross_cache, **progress)
            signs = recover_signs(query, extended, result.weights, result.groups,
                min_votes=config.sign_min_votes, confidence=config.sign_confidence,
                cache=cross_cache, **progress).signs
            np.savez(trial / 'signatures.npz', weights=result.weights, biases=result.biases, signs=signs)
            if not np.all(signs):
                item['reason'] = 'unresolved_cross_layer_signs'
                continue
            relative = (extended.calibrate_boundary_tolerances(result.weights, result.biases, result.groups,
                multiplier=config.boundary_error_multiplier, maximum=config.boundary_max_relative_tolerance,
                batch_size=config.batch_size, **progress) if config.boundary_filter == 'adaptive'
                else np.zeros(len(signs)))
            proposal = copy.deepcopy(extended)
            proposal.append_layer(result.weights * signs, result.biases * signs, layout='in_out')
            proposal.append_boundary_tolerances(relative)
            torch.save(proposal.state_dict(), trial / 'candidate.pth')
            np.savez(trial / 'cross_layer_parameters.npz',
                     ordinary_weights=(result.weights * signs)[:width],
                     cross_weights=(result.weights * signs)[width:], absorbed_bias=result.biases * signs)
            remaining, _, unexplained, _ = _active_remaining(proposal, state['pool'], quarantine, config,
                                                   'cross-layer candidate coverage')
            item['termination_coverage'] = termination_coverage(int(unexplained.sum()), len(state['pool']), config)
            if item['termination_coverage']['passed']:
                candidate, state['pool'], finish = _finish_candidate(query, proposal, state['pool'],
                    rng, validation_rng, config, trial, quarantine=quarantine, detection_rng=detection_rng,
                    retained_state=state)
                item['finish'] = finish
                if candidate is not None:
                    item['accepted'] = True
                    metadata.update(status='accepted', output_layer=True, accepted_through=layer + 1,
                                    finish=finish)
                    return candidate, metadata, []
                continue
            detection = detect_unknown_prefix(query, proposal, remaining, detection_rng, config)
            item.update(detection=detection, reason='cross_layer_detection_rejected')
            if not detection['passed']:
                continue
            known = {PointQuarantine.point_id(point) for point in state['pool']}
            extra = collect_points_unknown(query, rng, config, 'cross-layer independent confirmation',
                                           sweeps=config.confirmation_sweeps)
            state['pool'] = _merge_pool(state['pool'], extra)
            fresh = np.asarray([point for point in np.unique(extra, axis=0)
                                if PointQuarantine.point_id(point) not in known],
                               dtype=np.float64).reshape(-1, query.input_dim)
            fresh, _ = remaining_points(proposal, fresh, config, 'cross-layer confirmation coverage')
            np.save(trial / 'confirmation_points.npy', fresh)
            confirmation = detect_unknown_prefix(query, proposal, fresh, detection_rng, config)
            item.update(confirmation=confirmation, reason='cross_layer_confirmation_rejected')
            if confirmation['passed']:
                item.update(accepted=True, reason='cross_layer_confirmed')
                metadata.update(status='accepted', output_layer=False, accepted_through=layer + 1)
                return proposal, metadata, [(remaining, detection, 'cross_layer_detection'),
                                          (fresh, confirmation, 'cross_layer_confirmation')]
        metadata['status'] = 'rejected'
        return None, metadata, []
    finally:
        metadata['queries'] = query.query_count
        metadata['cache'] = cross_cache.summary()
        np.save(directory / 'point_pool.npy', state['pool'])
        save_json(directory / 'report.json', metadata)
