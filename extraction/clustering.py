"""Graph clustering, first-layer geometry splitting, and kernel-based merging."""
import numpy as np
from extraction.utils import GlobalConfig, MIN_SAME_SIZE, BLOCK_MULTIPLY_FACTOR
from utils.progress import Progress
import copy
from extraction.utils import compute_difference


def ratio_normalize(possible_matrix_rows, *, verbose=True, cache=None):
    """Normalize component signatures and reuse results for identical inputs."""
    key = cache.key((), 'ratio_normalize', possible_matrix_rows) if cache is not None else None
    result = cache.get(key) if cache is not None else None
    if result is None:
        result = _ratio_normalize(possible_matrix_rows, verbose=False)
        if cache is not None:
            cache.put(key, result)
    if verbose:
        print('Best column', result[1], 'with error', result[2])
    return result


def _ratio_normalize(possible_matrix_rows, *, verbose=True):
    """Normalize ratio rows containing NaNs.

    Returns the normalized ratios, reference column, and normalization error.
    """
    ratio_evidence = [
        [[] for _ in range(possible_matrix_rows.shape[1])]
        for _ in range(possible_matrix_rows.shape[1])
    ]

    for row in possible_matrix_rows:
        for i in range(len(row)):
            for j in range(len(row)):
                ratio_evidence[i][j].append(row[i] / row[j])

    ratio_evidence = np.array(ratio_evidence, dtype=np.float64)

    medians = np.nanmedian(ratio_evidence, axis=2)
    errors = (
        np.nanstd(ratio_evidence, axis=2)
        / np.sum(~np.isnan(ratio_evidence), axis=2) ** 0.5
    )
    errors += 1e-2 * (np.sum(~np.isnan(ratio_evidence), axis=2) == 1)
    errors /= np.abs(medians)
    errors[np.isnan(errors)] = 1e6

    ratio_evidence = medians

    nancount = np.sum(np.isnan(ratio_evidence), axis=0)
    column_ok = np.min(nancount) == nancount

    best = (None, np.inf)

    for column in range(len(column_ok)):
        if not column_ok[column]:
            continue

        # Match [:, column, :] without allocating a d**3 tensor.
        via_column = ratio_evidence[:, column, None] * ratio_evidence[column, :]
        quality = np.nansum(np.abs(via_column - ratio_evidence))
        if quality < best[1]:
            best = (column, quality)

    column, best_error = best
    if verbose:
        print("Best column", column, "with error", best_error)

    return ratio_evidence[:, column], column, best_error


# Shared ratio graph clustering; no bias or target structure.


def basic_cluster(all_ratios, block_size=128, *, progress=None):
    """Build a similarity graph over all reference coordinates and return its components.

    Union-find avoids duplicate edges; two-axis blocking bounds temporary memory.
    Zero and nonfinite coordinates provide no matching evidence. Singletons are retained.
    """
    ratios = np.asarray(all_ratios, dtype=np.float64)
    if ratios.ndim != 2 or ratios.shape[1] < 1 or block_size < 1:
        raise ValueError('Expected (n, d) ratios with d > 0 and positive block_size')
    n, d = ratios.shape
    completed_blocks = 0
    parent = list(range(n))
    sizes = [1] * n

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    tol = GlobalConfig.BLOCK_ERROR_TOL * np.log(d)
    threshold = d // 2 if d > 100 else max(
        MIN_SAME_SIZE, BLOCK_MULTIPLY_FACTOR * (np.log(d) - 2)
    )
    for dim in range(d):
        usable = np.isfinite(ratios[:, dim]) & (ratios[:, dim] != 0)
        scaled = np.full_like(ratios, np.nan)
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            scaled[usable] = ratios[usable] / ratios[usable, dim:dim + 1]
        for start in range(0, n, block_size):
            a = scaled[start:start + block_size, None, :]
            for other in range(start, n, block_size):
                b = scaled[None, other:other + block_size, :]
                valid = np.isfinite(a) & np.isfinite(b) & (a != 0) & (b != 0)
                with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                    diff = np.abs(a - b)
                    error = diff / np.abs(a) + diff / np.abs(b)
                matched = np.sum(valid & (error < tol), axis=2) >= threshold
                for i, j in zip(*np.nonzero(matched)):
                    i, j = int(i + start), int(j + other)
                    if i >= j:
                        continue
                    left, right = root(i), root(j)
                    if left != right:
                        if sizes[left] < sizes[right]:
                            left, right = right, left
                        parent[right] = left
                        sizes[left] += sizes[right]
                completed_blocks += 1
                if progress is not None:
                    progress.update(completed_blocks,
                                    f'axis={dim + 1}/{d} rows={start}:{min(start + block_size, n)} '
                                    f'cols={other}:{min(other + block_size, n)}')
    if progress is not None and n == 0:
        progress.update(0, 'no successful ratios; empty clustering')
    components = {}
    for i in range(n):
        components.setdefault(root(i), []).append(i)
    return sorted(components.values(), key=lambda c: (-len(c), c[0]))


# Build edges from sampled axes without reading target structure.


def source_graph(ratios, rng, *, first_layer, progress_mode='auto', progress_interval=5.0, cache=None):
    ratios = np.asarray(ratios, dtype=np.float64)
    n, d = ratios.shape
    parent = list(range(n))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    blocks = (n + 127) // 128
    total = sum((min(1000, n - start) + 127) // 128 * blocks * min(d, 20)
                for start in range(0, n, 1000))
    threshold = d // 2 if d > 100 else max(MIN_SAME_SIZE, BLOCK_MULTIPLY_FACTOR * (np.log(d) - 2))
    tolerance = GlobalConfig.BLOCK_ERROR_TOL * np.log(d)
    completed = 0
    with Progress('source ratio graph', total, progress_interval, mode=progress_mode) as progress:
        for batch in range(0, n, 1000):
            # Shuffle all axes for each 1,000-row batch and select 20 using the caller's RNG.
            axes = rng.permutation(d)[:20]
            for axis in axes:
                with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                    scaled = ratios / ratios[:, axis:axis + 1]
                for start in range(batch, min(batch + 1000, n), 128):
                    stop = min(start + 128, batch + 1000, n)
                    a = scaled[start:stop, None, :]
                    for other in range(0, n, 128):
                        b = scaled[None, other:other + 128, :]
                        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                            diff = np.abs(a - b)
                            error = diff / np.abs(a) + diff / np.abs(b)
                        matches = np.sum(error < tolerance, axis=2) >= threshold
                        for i, j in zip(*np.nonzero(matches)):
                            i, j = int(i + start), int(j + other)
                            # Exclude identical and adjacent indices with abs(i - j) > 1.
                            if abs(i - j) > 1:
                                left, right = root(i), root(j)
                                parent[right] = left
                        completed += 1
                        progress.update(completed, f'batch={batch // 1000 + 1} axis={axis} rows={start}:{stop}')
    components = {}
    for i in range(n):
        components.setdefault(root(i), []).append(i)
    groups = sorted(components.values(), key=lambda g: (-len(g), g[0]))
    # Return singletons separately for kernel merging and unexplained-point tracking.
    singles = [g for g in groups if len(g) == 1]
    groups = [g for g in groups if len(g) > 1]
    if not first_layer:
        while groups:
            with Progress('source component normalization', len(groups), progress_interval,
                          mode=progress_mode) as progress:
                rows = []
                for i, group in enumerate(groups):
                    rows.append(ratio_normalize(ratios[group], verbose=False, cache=cache)[0])
                    progress.update(i + 1)
            count = len(rows)
            blocks = (count + 127) // 128
            with Progress('source component graph', d * blocks * (blocks + 1) // 2,
                          progress_interval, mode=progress_mode) as progress:
                merged = basic_cluster(rows, progress=progress)
            updated = [sum((groups[i] for i in component), []) for component in merged]
            if len(updated) >= len(groups):
                break
            groups = updated
    return groups + singles


# Split first-layer graph components by full hyperplane consistency.
# Split groups using query-derived ratios and witnesses.
# This does not establish that a boundary belongs to the first layer.


def split_first_layer_groups(ratios, points, groups, *, direction_tolerance=1e-4,
                             bias_tolerance=1e-5, diagnostics=None,
                             progress_mode='auto', progress_interval=5.0):
    for value in (direction_tolerance, bias_tolerance):
        if not np.isfinite(value) or value <= 0:
            raise ValueError('First-layer geometry tolerances must be finite and positive')
    ratios = np.asarray(ratios, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64).reshape(ratios.shape)
    # Scale first to avoid overflow and keep invariance under nonzero rescaling.
    scale = np.max(np.abs(ratios), axis=1)
    valid = np.isfinite(ratios).all(axis=1) & np.isfinite(points).all(axis=1) & (scale > 0)
    unit = np.zeros_like(ratios)
    unit[valid] = ratios[valid] / scale[valid, None]
    unit[valid] /= np.linalg.norm(unit[valid], axis=1)[:, None]
    bias = -np.einsum('ij,ij->i', unit, points)
    valid &= np.isfinite(bias)
    result, mapping = [], []
    completed = 0
    with Progress('first-layer hyperplane consistency', sum(map(len, groups)),
                  progress_interval, mode=progress_mode) as progress:
        for group in groups:
            parts = []
            for index in sorted(group):
                index = int(index)
                destination = None
                if valid[index]:
                    for part in parts:
                        if not np.all(valid[part]):
                            continue
                        alignment = np.where(unit[part] @ unit[index] >= 0, 1., -1.)
                        distance = np.linalg.norm(unit[part] * alignment[:, None] - unit[index], axis=1)
                        offsets = np.abs(bias[part] * alignment - bias[index])
                        # Complete-link check: a chain of near matches cannot
                        # bridge endpoints that disagree. No target parameters.
                        if np.all(distance <= direction_tolerance) and np.all(offsets <= bias_tolerance):
                            destination = part
                            break
                if destination is None:
                    parts.append([index])
                else:
                    destination.append(index)
                completed += 1
                progress.update(completed, f'groups={len(result) + len(parts)}')
            mapping.append(list(range(len(result), len(result) + len(parts))))
            result.extend(parts)
    if diagnostics is not None:
        diagnostics.update(input_groups=len(groups), output_groups=len(result),
            direction_tolerance=direction_tolerance, bias_tolerance=bias_tolerance,
            invalid_rows=int((~valid).sum()), split_groups=sum(len(x) > 1 for x in mapping),
            input_to_output_groups=mapping, groups=result,
            interpretation='local hyperplane consistency; not layer membership')
    return result


# Licensed under MIT.
# args.progress_iter adapts tqdm to utils.progress; model is unused.

def merge_bad_critical_points_into_components(args, model, critical_points, special_sols, ker_spaces, bad_indices, components, partial_rows):
    all_criticals = [cp.point for cp in critical_points]
    all_h_matrices = [cp.h_matrix for cp in critical_points]
    all_criticals = np.array(all_criticals, dtype=np.float64)
    merged_indices = []
    merged_components = copy.deepcopy(components)
    merged_partial_rows = copy.deepcopy(partial_rows)
    for i in args.progress_iter(range(len(bad_indices)), desc='merge bad critical points with components'):
        if i in merged_indices:
            continue
        bad_critical_point = all_criticals[bad_indices[i]]
        bad_h_matrix = all_h_matrices[bad_indices[i]]
        bad_special_sol = special_sols[bad_indices[i]]
        bad_ker_space = ker_spaces[bad_indices[i]]
        column_is_zero = np.mean(np.abs(bad_h_matrix) < 1e-08, axis=0) > 0.5
        column_non_zero = ~column_is_zero
        ker_size = np.sum(column_non_zero) - np.linalg.matrix_rank(bad_h_matrix)
        for j in range(len(merged_components)):
            recovered_weight_ratio = merged_partial_rows[j].copy()
            recovered_weight_non_zero = ~np.isnan(recovered_weight_ratio)
            recovered_weight_ratio_nonan = recovered_weight_ratio.copy()
            recovered_weight_ratio_nonan[np.isnan(recovered_weight_ratio_nonan)] = 0
            good_weight = np.array(recovered_weight_ratio_nonan).reshape(-1, 1)
            column_non_zero_union = recovered_weight_non_zero & column_non_zero
            if 0 + ker_size + 2 <= np.sum(column_non_zero_union):
                merge_matrix = np.hstack((-np.array(bad_ker_space), -np.array(bad_special_sol).reshape(-1, 1)))
                merge_matrix = merge_matrix[column_non_zero_union]
                merge_matrix = merge_matrix[:0 + ker_size + 1, :]
                if np.linalg.matrix_rank(merge_matrix) < ker_size + 1:
                    continue
                merge_y = -good_weight
                merge_y = merge_y[column_non_zero_union]
                merge_y = merge_y[:0 + ker_size + 1]
                merge_sol = np.linalg.lstsq(merge_matrix, merge_y)
                merge_coefficient = merge_sol[0]
                weight_ratio = np.array(bad_special_sol).reshape(-1, 1)
                for k in range(ker_size):
                    weight_ratio += merge_coefficient[k] / merge_coefficient[-1] * bad_ker_space[:, k].reshape(-1, 1)
                weight_ratio_multiply_k = weight_ratio * merge_coefficient[-1]
                difference, _, _, _ = compute_difference(good_weight, weight_ratio_multiply_k)
                if args.dataset == 'mnist':
                    if args.real_attack == 0:
                        threshold = 1e-05
                    else:
                        threshold = 1e-06
                elif args.dataset == 'cifar10':
                    threshold = 1e-06
                if difference < threshold:
                    merged_components[j].append(bad_indices[i])
                    merged_indices.append(i)
                    new_weight_indices = column_non_zero.astype(int) - recovered_weight_non_zero.astype(int) > 0
                    for l in range(len(new_weight_indices)):
                        if new_weight_indices[l]:
                            if np.isnan(merged_partial_rows[j][l]):
                                # The least-squares result is a (d, 1) column;
                                # partial rows store scalar coordinates.
                                merged_partial_rows[j][l] = weight_ratio_multiply_k[l, 0]
                    break
    return (merged_components, merged_partial_rows, merged_indices)

def merge_bad_critical_points_with_each_other(args, model, critical_points, special_sols, ker_spaces, bad_indices_not_merged, good_indices_not_merged, components, partial_rows):
    all_criticals = [cp.point for cp in critical_points]
    all_h_matrices = [cp.h_matrix for cp in critical_points]
    all_criticals = np.array(all_criticals, dtype=np.float64)
    merged_indices = []
    merged_components = copy.deepcopy(components)
    merged_partial_rows = copy.deepcopy(partial_rows)
    not_merged_indices = copy.deepcopy(bad_indices_not_merged)
    not_merged_indices.extend(good_indices_not_merged)
    for i in args.progress_iter(range(0, len(bad_indices_not_merged)), desc='merge bad critical points with each other'):
        if i in merged_indices:
            continue
        new_partial_weight = []
        new_merged_critical_points = []
        bad_critical_point_i = all_criticals[bad_indices_not_merged[i]]
        bad_h_matrix_i = all_h_matrices[bad_indices_not_merged[i]]
        bad_special_sol_i = special_sols[bad_indices_not_merged[i]]
        bad_ker_space_i = ker_spaces[bad_indices_not_merged[i]]
        column_is_zero_i = np.mean(np.abs(bad_h_matrix_i) < 1e-08, axis=0) > 0.5
        column_non_zero_i = ~column_is_zero_i
        ker_size_i = np.sum(column_non_zero_i) - np.linalg.matrix_rank(bad_h_matrix_i)
        first_merged_j = True
        for j in range(i + 1, len(not_merged_indices)):
            if j in merged_indices:
                continue
            bad_critical_point_j = all_criticals[not_merged_indices[j]]
            bad_h_matrix_j = all_h_matrices[not_merged_indices[j]]
            bad_special_sol_j = special_sols[not_merged_indices[j]]
            bad_ker_space_j = ker_spaces[not_merged_indices[j]]
            column_is_zero_j = np.mean(np.abs(bad_h_matrix_j) < 1e-08, axis=0) > 0.5
            column_non_zero_j = ~column_is_zero_j
            ker_size_j = np.sum(column_non_zero_j) - np.linalg.matrix_rank(bad_h_matrix_j)
            column_non_zero_union = column_non_zero_i & column_non_zero_j
            if ker_size_i + ker_size_j + 2 <= np.sum(column_non_zero_union):
                if ker_size_i != 0 and ker_size_j == 0:
                    merge_matrix = np.hstack((np.array(bad_ker_space_i), -np.array(bad_special_sol_j).reshape(-1, 1)))
                elif ker_size_i != 0 and ker_size_j != 0:
                    merge_matrix = np.hstack((np.array(bad_ker_space_i), -np.array(bad_ker_space_j), -np.array(bad_special_sol_j).reshape(-1, 1)))
                merge_matrix = merge_matrix[column_non_zero_union]
                merge_matrix = merge_matrix[:ker_size_i + ker_size_j + 1, :]
                if np.linalg.matrix_rank(merge_matrix) < ker_size_i + ker_size_j + 1:
                    continue
                det_merge_matrix = np.linalg.det(merge_matrix)
                if abs(det_merge_matrix) < 1e-06:
                    continue
                merge_y = -np.array(bad_special_sol_i).reshape(-1, 1)
                merge_y = merge_y[column_non_zero_union]
                merge_y = merge_y[:ker_size_i + ker_size_j + 1]
                merge_sol = np.linalg.lstsq(merge_matrix, merge_y)
                merge_coefficient = merge_sol[0]
                weight_ratio_i = np.array(bad_special_sol_i).reshape(-1, 1)
                for k in range(ker_size_i):
                    weight_ratio_i += merge_coefficient[k] * bad_ker_space_i[:, k].reshape(-1, 1)
                weight_ratio_j = np.array(bad_special_sol_j).reshape(-1, 1)
                for k in range(ker_size_j):
                    weight_ratio_j += merge_coefficient[ker_size_i + k] / merge_coefficient[-1] * bad_ker_space_j[:, k].reshape(-1, 1)
                weight_ratio_j_multiply_k = weight_ratio_j * merge_coefficient[-1]
                difference, _, _, _ = compute_difference(weight_ratio_i, weight_ratio_j_multiply_k)
                if args.real_attack == 0:
                    threshold = 1e-06
                else:
                    threshold = 1e-07
                if difference < threshold:
                    if first_merged_j:
                        merged_indices.append(i)
                        new_merged_critical_points.append(not_merged_indices[i])
                        new_partial_weight = weight_ratio_i.copy()
                        first_merged_j = False
                    merged_indices.append(j)
                    new_merged_critical_points.append(not_merged_indices[j])
                    new_partial_weight = new_partial_weight.flatten()
                    new_partial_weight_non_zero = ~np.isnan(new_partial_weight)
                    new_weight_indices = column_non_zero_j.astype(int) - new_partial_weight_non_zero.astype(int) > 0
                    same_indices = np.where(new_partial_weight_non_zero & column_non_zero_j)[0]
                    common_index = same_indices[0]
                    factor = new_partial_weight[common_index] / weight_ratio_j_multiply_k[common_index, 0]
                    for l in range(len(new_weight_indices)):
                        if new_weight_indices[l]:
                            if np.isnan(new_partial_weight[l]):
                                new_partial_weight[l] = weight_ratio_j_multiply_k[l, 0] * factor
        if len(new_partial_weight) > 0:
            merged = False
            for j, partial_weight in enumerate(merged_partial_rows):
                new_partial_weight = new_partial_weight.flatten()
                difference, _, _, _ = compute_difference(partial_weight, new_partial_weight)
                if difference == 0.0:
                    continue
                elif difference < 1e-05:
                    partial_weight_column_non_zero = ~np.isnan(partial_weight)
                    new_partial_weight_non_zero = ~np.isnan(new_partial_weight)
                    new_weight_indices = partial_weight_column_non_zero.astype(int) - new_partial_weight_non_zero.astype(int) > 0
                    same_indices = np.where(partial_weight_column_non_zero & new_partial_weight_non_zero)[0]
                    common_index = same_indices[0]
                    factor = new_partial_weight[common_index] / partial_weight[common_index]
                    for l in range(len(new_weight_indices)):
                        if new_weight_indices[l]:
                            new_partial_weight[l] = partial_weight[l] * factor
                    merged_partial_rows[j] = new_partial_weight
                    merged_components[j].extend(new_merged_critical_points)
                    merged = True
                    break
            if not merged:
                merged_partial_rows = np.vstack((merged_partial_rows, new_partial_weight))
                merged_components.extend([new_merged_critical_points])
    return (merged_components, merged_partial_rows, merged_indices)

def merge_bad_critical_points_into_a_specific_component(args, model, critical_points, special_sols, ker_spaces, bad_indices, component, partial_row):
    all_criticals = [cp.point for cp in critical_points]
    all_h_matrices = [cp.h_matrix for cp in critical_points]
    all_criticals = np.array(all_criticals, dtype=np.float64)
    merged_indices = []
    merged_component = copy.deepcopy(component)
    merged_partial_row = copy.deepcopy(partial_row)
    weak_indices = []
    while True:
        weak_len_before = len(weak_indices)
        for i in range(len(bad_indices)):
            if i in merged_indices:
                continue
            bad_critical_point = all_criticals[bad_indices[i]]
            bad_h_matrix = all_h_matrices[bad_indices[i]]
            bad_special_sol = special_sols[bad_indices[i]]
            bad_ker_space = ker_spaces[bad_indices[i]]
            column_is_zero = np.mean(np.abs(bad_h_matrix) < 1e-08, axis=0) > 0.5
            column_non_zero = ~column_is_zero
            ker_size = np.sum(column_non_zero) - np.linalg.matrix_rank(bad_h_matrix)
            recovered_weight_ratio = merged_partial_row.copy()
            recovered_weight_non_zero = ~np.isnan(recovered_weight_ratio)
            recovered_weight_ratio_nonan = recovered_weight_ratio.copy()
            recovered_weight_ratio_nonan[np.isnan(recovered_weight_ratio_nonan)] = 0
            good_weight = np.array(recovered_weight_ratio_nonan).reshape(-1, 1)
            column_non_zero_union = recovered_weight_non_zero & column_non_zero
            if 0 + ker_size + 2 <= np.sum(column_non_zero_union):
                merge_matrix = np.hstack((-np.array(bad_ker_space), -np.array(bad_special_sol).reshape(-1, 1)))
                merge_matrix = merge_matrix[column_non_zero_union]
                merge_matrix = merge_matrix[:0 + ker_size + 1, :]
                if np.linalg.matrix_rank(merge_matrix) < ker_size + 1:
                    continue
                merge_y = -good_weight
                merge_y = merge_y[column_non_zero_union]
                merge_y = merge_y[:0 + ker_size + 1]
                merge_sol = np.linalg.lstsq(merge_matrix, merge_y)
                merge_coefficient = merge_sol[0]
                weight_ratio = np.array(bad_special_sol).reshape(-1, 1)
                for k in range(ker_size):
                    weight_ratio += merge_coefficient[k] / merge_coefficient[-1] * bad_ker_space[:, k].reshape(-1, 1)
                weight_ratio_multiply_k = weight_ratio * merge_coefficient[-1]
                difference, _, _, _ = compute_difference(good_weight, weight_ratio_multiply_k)
                if difference < 1e-06:
                    merged_component.append(bad_indices[i])
                    merged_indices.append(i)
                    new_weight_indices = column_non_zero.astype(int) - recovered_weight_non_zero.astype(int) > 0
                    for l in range(len(new_weight_indices)):
                        if new_weight_indices[l]:
                            if np.isnan(merged_partial_row[l]):
                                merged_partial_row[l] = weight_ratio_multiply_k[l, 0]
            elif 0 + ker_size + 1 <= np.sum(column_non_zero_union):
                if i not in weak_indices:
                    weak_indices.append(i)
        weak_len_after = len(weak_indices)
        if weak_len_after - weak_len_before == 0:
            break
        possible_weight_flag = [False] * len(merged_partial_row)
        possible_weight = [[] for _ in range(len(merged_partial_row))]
        for i in range(len(merged_partial_row)):
            if not np.isnan(merged_partial_row[i]):
                possible_weight_flag[i] = True
                possible_weight[i] = [merged_partial_row[i], 1]
        for i in weak_indices:
            bad_critical_point = all_criticals[bad_indices[i]]
            bad_h_matrix = all_h_matrices[bad_indices[i]]
            bad_special_sol = special_sols[bad_indices[i]]
            bad_ker_space = ker_spaces[bad_indices[i]]
            column_is_zero = np.mean(np.abs(bad_h_matrix) < 1e-08, axis=0) > 0.5
            column_non_zero = ~column_is_zero
            ker_size = np.sum(column_non_zero) - np.linalg.matrix_rank(bad_h_matrix)
            recovered_weight_ratio = merged_partial_row.copy()
            recovered_weight_non_zero = ~np.isnan(recovered_weight_ratio)
            recovered_weight_ratio_nonan = recovered_weight_ratio.copy()
            recovered_weight_ratio_nonan[np.isnan(recovered_weight_ratio_nonan)] = 0
            good_weight = np.array(recovered_weight_ratio_nonan).reshape(-1, 1)
            column_non_zero_union = recovered_weight_non_zero & column_non_zero
            merge_matrix = np.hstack((-np.array(bad_ker_space), -np.array(bad_special_sol).reshape(-1, 1)))
            merge_matrix = merge_matrix[column_non_zero_union]
            merge_matrix = merge_matrix[:0 + ker_size + 1, :]
            if np.linalg.matrix_rank(merge_matrix) < ker_size + 1:
                continue
            merge_y = -good_weight
            merge_y = merge_y[column_non_zero_union]
            merge_y = merge_y[:0 + ker_size + 1]
            merge_sol = np.linalg.lstsq(merge_matrix, merge_y)
            merge_coefficient = merge_sol[0]
            weight_ratio = np.array(bad_special_sol).reshape(-1, 1)
            for k in range(ker_size):
                weight_ratio += merge_coefficient[k] / merge_coefficient[-1] * bad_ker_space[:, k].reshape(-1, 1)
            weight_ratio_multiply_k = weight_ratio * merge_coefficient[-1]
            for l in range(len(possible_weight_flag)):
                if not possible_weight_flag[l]:
                    if not np.isnan(weight_ratio_multiply_k[l, 0]):
                        if len(possible_weight[l]) == 0:
                            possible_weight[l].append([weight_ratio_multiply_k[l, 0], 1])
                        else:
                            for m in range(len(possible_weight[l])):
                                if abs(possible_weight[l][m][0] - weight_ratio_multiply_k[l, 0]) < 0.0001:
                                    possible_weight[l][m][1] += 1
                                    break
                                elif m == len(possible_weight[l]) - 1:
                                    possible_weight[l].append([weight_ratio_multiply_k[l, 0], 1])
        for i in range(len(possible_weight_flag)):
            if not possible_weight_flag[i]:
                if len(possible_weight[i]) > 0:
                    max_count = 0
                    max_index = 0
                    for j in range(len(possible_weight[i])):
                        if possible_weight[i][j][1] > max_count:
                            max_count = possible_weight[i][j][1]
                            max_index = j
                    if max_count > 1:
                        merged_partial_row[i] = possible_weight[i][max_index][0]
    return (merged_component, merged_partial_row, merged_indices)
