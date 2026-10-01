"""Configure, sample, validate, and recover networks with unknown hidden architectures."""
from extraction.utils import query_numpy, RecoveryCache
from argparse import Namespace
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import numpy as np
import torch
from models.base import RecoveryModel
from models.base import QueryBudgetExceeded
from extraction.search import do_better_sweep
from extraction.signature_recovery import InsufficientEvidence
from extraction.precision import improve_precision
from extraction.follow import get_more_crit_pts
from extraction.sign_recovery import recover_signs, recover_last_hidden_signs
from extraction.signature_recovery import recover_output_layer
from utils.progress import Progress, log_progress
from utils.paths import runtime_path
from dataclasses import replace
import copy
import hashlib
from extraction.signature_recovery import recover_layer_weights_unknown
from extraction.precision import refine_first_layer_signatures, refine_deep_layer_signatures, validate_refinement_options
from extraction.error_detection import check_point, check_linear_output_system, count_directions


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


@dataclass
class UnknownExtractionConfig:
    """Configure extraction using only input and output dimensions.

    Hidden depths and widths are inferred during recovery.
    """
    seed: int = 42
    points_per_round: int = 2000
    max_rounds: int = 3
    max_sweeps: int = 1000
    sweep_range: float = 1000.0
    output_index: int = 0
    ratio_eps: float = 1e-5
    cluster_tolerance: float = 1e-4
    cluster_method: str = 'source'
    first_layer_clustering: str = 'geometry'
    first_layer_direction_tolerance: float = 1e-4
    first_layer_bias_tolerance: float = 1e-5
    min_witnesses: int = 3
    bad_pair_budget: int = 100
    sign_method: str = 'neuron-wiggle'
    sign_min_votes: int = 1
    sign_confidence: float = 0.5
    follow_attempts: int = 10
    follow_step: float = 0.1
    follow_grad_eps: float = 1e-4
    improve_precision: bool = True
    precision_rank_tolerance: float = 1e-5
    output_samples: int = 0
    validation_samples: int = 1000
    batch_size: int = 256
    max_queries: int | None = None
    progress_mode: str = 'auto'
    progress_interval: float = 5.0
    # Absolute tolerance for output fitting and final logits validation.
    validation_tolerance: float = 2e-3
    initial_points: int | None = None
    additional_points: int | None = None
    max_hidden_layers: int = 32
    confirmation_sweeps: int = 8
    boundary_filter: str = 'adaptive'
    boundary_tolerance: float = 1e-5
    boundary_error_multiplier: float = 2.0
    boundary_max_relative_tolerance: float = 1e-6
    termination_max_remaining_rate: float = 0.02
    detection_max_inconsistent_rate: float = 0.01
    detection_max_skipped_rate: float = 0.01
    detection_min_tested: int = 100
    quarantine_min_confirmations: int = 2
    quarantine_max_failures: int = 3
    signature_refinement: str = 'stable'
    refinement_direction_factor: int = 2
    refinement_max_witnesses: int = 8
    refinement_step_trials: int = 3
    refinement_tolerance: float = 1e-4
    defer_incomplete_after_rounds: int = 2
    persistent_after_rounds: int = 2
    persistent_max_rounds: int = 3
    persistent_match_tolerance: float = 1e-4
    persistent_constraint_tolerance: float = 1e-5
    persistent_pair_budget: int = 512
    persistent_max_points: int = 64

    def __post_init__(self):
        # Retain points_per_round for callers using the previous configuration API.
        if self.initial_points is None:
            self.initial_points = self.points_per_round
        if self.additional_points is None:
            self.additional_points = self.initial_points

    def validate_dimensions(self, input_dim, output_dim):
        if (isinstance(self.quarantine_max_failures, bool)
                or not isinstance(self.quarantine_max_failures, int) or self.quarantine_max_failures <= 0):
            raise ValueError('quarantine_max_failures must be a positive integer')
        if self.first_layer_clustering not in ('geometry', 'source'):
            raise ValueError('first_layer_clustering must be geometry or source')
        for value in (self.first_layer_direction_tolerance, self.first_layer_bias_tolerance):
            if not np.isfinite(value) or value <= 0:
                raise ValueError('First-layer geometry tolerances must be finite and positive')
        for name in ('points_per_round', 'max_rounds', 'max_sweeps', 'min_witnesses',
                     'sign_min_votes', 'validation_samples', 'batch_size'):
            if getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive')
        if self.max_queries is not None and (isinstance(self.max_queries, bool) or
                not isinstance(self.max_queries, int) or self.max_queries <= 0):
            raise ValueError('max_queries must be a positive integer or None (unlimited)')
        if self.seed < 0 or self.follow_attempts < 0 or self.output_samples < 0 or self.bad_pair_budget < 0:
            raise ValueError('seed, follow_attempts, output_samples and bad_pair_budget must be nonnegative')
        for name in ('sweep_range', 'ratio_eps', 'cluster_tolerance', 'follow_step', 'follow_grad_eps',
                     'precision_rank_tolerance', 'validation_tolerance', 'progress_interval'):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if not 0 <= self.output_index < output_dim:
            raise ValueError('output_index outside logits')
        if not 0.5 <= self.sign_confidence <= 1:
            raise ValueError('sign_confidence must be in (0.5, 1]')
        if self.sign_method not in ('neuron-wiggle', 'legacy-last-hidden-system'):
            raise ValueError('Unknown sign method')
        if self.cluster_method not in ('source', 'basic'):
            raise ValueError('cluster_method must be source or basic')
        if self.progress_mode not in ('auto', 'bar', 'log', 'off'):
            raise ValueError('Unknown progress mode')
        if (isinstance(self.persistent_after_rounds, bool)
                or not isinstance(self.persistent_after_rounds, int) or self.persistent_after_rounds < 0):
            raise ValueError('persistent_after_rounds must be a nonnegative integer (0 disables)')
        if (isinstance(self.persistent_max_rounds, bool)
                or not isinstance(self.persistent_max_rounds, int) or self.persistent_max_rounds <= 0):
            raise ValueError('persistent_max_rounds must be a positive integer')
        if not np.isfinite(self.persistent_match_tolerance) or self.persistent_match_tolerance <= 0:
            raise ValueError('persistent_match_tolerance must be finite and positive')
        if (not np.isfinite(self.persistent_constraint_tolerance)
                or not 0 < self.persistent_constraint_tolerance < 1):
            raise ValueError('persistent_constraint_tolerance must be finite and in (0, 1)')
        if (isinstance(self.persistent_pair_budget, bool)
                or not isinstance(self.persistent_pair_budget, int) or self.persistent_pair_budget <= 0):
            raise ValueError('persistent_pair_budget must be a positive integer')
        if (isinstance(self.persistent_max_points, bool)
                or not isinstance(self.persistent_max_points, int)
                or self.persistent_max_points < max(2, self.min_witnesses)):
            raise ValueError('persistent_max_points must be an integer >= max(2, min_witnesses)')
        if (isinstance(self.defer_incomplete_after_rounds, bool)
                or not isinstance(self.defer_incomplete_after_rounds, int)
                or self.defer_incomplete_after_rounds < 0):
            raise ValueError('defer_incomplete_after_rounds must be a nonnegative integer (0 disables)')
        if self.signature_refinement not in ('source', 'stable'):
            raise ValueError('signature_refinement must be source or stable')
        validate_refinement_options(self.refinement_direction_factor, self.refinement_max_witnesses,
                                    self.refinement_step_trials, self.refinement_tolerance, self.ratio_eps)
        for name in ('initial_points', 'additional_points'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if self.boundary_filter not in ('strict', 'adaptive'):
            raise ValueError('boundary_filter must be strict or adaptive')
        for name in ('boundary_tolerance', 'boundary_error_multiplier'):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if not np.isfinite(self.boundary_max_relative_tolerance) or self.boundary_max_relative_tolerance < 0:
            raise ValueError('boundary_max_relative_tolerance must be finite and nonnegative')
        for name in ('detection_max_inconsistent_rate', 'detection_max_skipped_rate',
                     'termination_max_remaining_rate'):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or not 0 <= value < 1:
                raise ValueError(f'{name} must be finite and in [0, 1)')
        for name in ('max_hidden_layers', 'confirmation_sweeps', 'detection_min_tested',
                     'quarantine_min_confirmations'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f'{name} must be a positive integer safety budget')


def collect_points_unknown(query, rng, config, label, *, sweeps=None, point_count=None):
    """Sample within a fixed budget; return an empty pool when no kinks are found.

    An empty pool does not establish successful recovery.
    """
    points = []
    point_count = config.additional_points if point_count is None else point_count
    limit = config.max_sweeps if sweeps is None else sweeps
    args = Namespace(dataset='cifar10', output_index=config.output_index)
    completed_sweeps = 0
    log_progress(f'{label}: target={point_count} max_sweeps={limit}')
    with Progress(label, limit, config.progress_interval, mode=config.progress_mode) as bar:
        for index in range(limit):
            found = do_better_sweep(
                args, query, offset=rng.normal(size=query.input_dim),
                direction=rng.normal(size=query.input_dim), low=-config.sweep_range,
                high=config.sweep_range, query=query, input_dim=query.input_dim)
            points.extend(found[:max(0, point_count - len(points))])
            completed_sweeps = index + 1
            bar.update(completed_sweeps, f'points={len(points)}/{point_count} queries={query.query_count}')
            if len(points) >= point_count:
                break
    reason = 'target_reached' if len(points) >= point_count else 'sweep_budget_exhausted'
    log_progress(f'{label}: collected={len(points)} target={point_count} '
                 f'sweeps={completed_sweeps}/{limit} stop={reason}')
    return np.asarray(points, dtype=np.float64).reshape(-1, query.input_dim)


def remaining_points(prefix, points, config, label):
    """Mark points explained by prefix boundaries.

    Active dimensions, measurement failures, and clustering thresholds do not remove evidence.
    """
    keep = np.ones(len(points), dtype=bool)
    with Progress(label, len(points), config.progress_interval, mode=config.progress_mode) as bar:
        for start in range(0, len(points), config.batch_size):
            stop = min(start + config.batch_size, len(points))
            adaptive = config.boundary_filter == 'adaptive'
            keep[start:stop] = ~prefix.prefix_boundary_mask(
                points[start:stop], tolerance=config.boundary_tolerance, normalize=adaptive
            ).cpu().numpy()
            bar.update(stop, f'explained={int((~keep[:stop]).sum())} remaining={int(keep[:stop].sum())}')
    return points[keep], keep


def detect_unknown_prefix(query, prefix, points, rng, config):
    """Accept a prefix based on mismatch and skipped rates and a minimum tested count."""
    details = []
    with Progress(f'layer={len(prefix.fcs)} error detection', len(points),
                  config.progress_interval, mode=config.progress_mode) as bar:
        failed = skipped = 0
        for index, point in enumerate(points):
            if not np.isfinite(point).all():
                record = dict(index=index, status='skipped', reason='nonfinite_point')
            elif count_directions(prefix, point) == 0:
                record = dict(index=index, status='skipped', reason='no_nonzero_prefix_values')
            else:
                result = check_point(query.tensor_forward, prefix, point, rng=rng,
                                     out_index=config.output_index, device=query.device, dtype=query.dtype)
                status = ('skipped' if result.consistent is None else
                          'consistent' if result.consistent else 'inconsistent')
                record = dict(index=index, status=status, rank=result.rank,
                              rank_augmented=result.rank_augmented, directions=result.direction_count)
            failed += record['status'] == 'inconsistent'
            skipped += record['status'] == 'skipped'
            details.append(record)
            bar.update(index + 1, f'tested={index + 1 - skipped} inconsistent={failed} skipped={skipped}')
    tested = len(points) - skipped
    inconsistent_rate = failed / tested if tested else None
    skipped_rate = skipped / len(points) if len(points) else None
    enough = tested >= config.detection_min_tested
    passed = (enough and inconsistent_rate <= config.detection_max_inconsistent_rate
              and skipped_rate <= config.detection_max_skipped_rate)
    return dict(passed=bool(passed), total=len(points), tested=tested, inconsistent=failed,
                skipped=skipped, inconsistent_rate=inconsistent_rate, skipped_rate=skipped_rate,
                sufficient_tests=enough, tolerated=bool(passed and (failed or skipped)),
                thresholds=dict(max_inconsistent_rate=config.detection_max_inconsistent_rate,
                                max_skipped_rate=config.detection_max_skipped_rate,
                                min_tested=config.detection_min_tested), points=details)


class PointQuarantine:
    """Track quarantined points and their history.

    Points discarded after repeated failed checks no longer participate in recovery or acceptance.
    """

    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.entries = {}
        self.prefixes = {}

    @staticmethod
    def point_id(point):
        point = np.asarray(point, dtype='<f8').copy()
        point[point == 0] = 0  # np.unique treats +0 and -0 as the same point.
        return hashlib.sha256(point.tobytes()).hexdigest()

    def mask(self, points):
        active = {key for key, entry in self.entries.items()
                  if entry['state'] in ('quarantined', 'discarded')}
        if not active:
            return np.zeros(len(points), dtype=bool)
        return np.fromiter((self.point_id(p) in active for p in points), dtype=bool, count=len(points))

    def discarded_mask(self, points):
        discarded = {key for key, entry in self.entries.items() if entry['state'] == 'discarded'}
        return np.fromiter((self.point_id(p) in discarded for p in points), dtype=bool, count=len(points))

    def points(self, *, include_discarded=True):
        return np.asarray([entry['point'] for entry in self.entries.values()
                           if include_discarded or entry['state'] != 'discarded'],
                          dtype=np.float64).reshape(-1, self.input_dim)

    def summary(self):
        active = sum(entry['state'] == 'quarantined' for entry in self.entries.values())
        discarded = sum(entry['state'] == 'discarded' for entry in self.entries.values())
        return dict(total=len(self.entries), active=active,
                    released=len(self.entries) - active - discarded, discarded=discarded)

    def add(self, prefix, points, detection, *, layer, round_number, stage):
        # Called only after both primary and independent confirmation have passed.
        added = 0
        for result in detection.get('points', []):
            if result['status'] == 'consistent':
                continue
            point = points[result['index']]
            key = self.point_id(point)
            if key in self.entries and self.entries[key]['state'] == 'discarded':
                continue
            entry = self.entries.setdefault(key, dict(point=point.tolist(), state='released',
                                                       confirmations=0, history=[]))
            entry.update(state='quarantined', origin_layer=layer, confirmations=0, failures=0)
            entry['history'].append(dict(stage=stage, layer=layer, round=round_number,
                                         event='quarantined', **result))
            added += 1
        if added:
            self.prefixes[layer] = copy.deepcopy(prefix)
        return added

    def recheck(self, query, rng, config, *, stage, passes=1):
        """Recheck with the prefix that triggered quarantine.

        Later layers must not hide errors in an earlier prefix.
        """
        records = []
        for repeat in range(passes):
            layers = sorted({entry['origin_layer'] for entry in self.entries.values()
                             if entry['state'] == 'quarantined'})
            for layer in layers:
                items = [(key, entry) for key, entry in self.entries.items()
                         if entry['state'] == 'quarantined' and entry['origin_layer'] == layer]
                points = np.asarray([entry['point'] for _, entry in items], dtype=np.float64)
                log_progress(f'Quarantine review: {stage}; origin_layer={layer} points={len(points)}')
                detection = detect_unknown_prefix(query, self.prefixes[layer], points, rng, config)
                for result in detection['points']:
                    key, entry = items[result['index']]
                    entry['confirmations'] = (entry['confirmations'] + 1
                                              if result['status'] == 'consistent' else 0)
                    entry['failures'] = (0 if result['status'] == 'consistent'
                                         else entry.get('failures', 0) + 1)
                    if entry['confirmations'] >= config.quarantine_min_confirmations:
                        entry['state'] = 'released'
                    elif entry['failures'] >= config.quarantine_max_failures:
                        entry['state'] = 'discarded'
                    event = dict(stage=stage, repeat=repeat + 1, layer=layer,
                                 event='recheck', state=entry['state'], failures=entry['failures'], **result)
                    entry['history'].append(event)
                    records.append(dict(point_id=key, **event))
        return dict(**self.summary(), checks=records)

    def save(self, directory):
        points = self.points()
        np.save(directory / 'quarantine_points.npy', points)
        save_json(directory / 'quarantine.json', dict(**self.summary(), prefixes={
            str(layer): dict(structure=list(prefix.structure), with_relu=prefix.with_relu,
                             boundary_profile=prefix.boundary_filter_profile())
            for layer, prefix in self.prefixes.items()}, points=[
            dict(point_id=key, point_index=index, **{k: v for k, v in entry.items() if k != 'point'})
            for index, (key, entry) in enumerate(self.entries.items())]))


def _active_remaining(prefix, pool, quarantine, config, label):
    """Return recovery points with separate unexplained and quarantine masks."""
    _, unexplained = remaining_points(prefix, pool, config, label)
    unexplained &= ~quarantine.discarded_mask(pool)
    isolated = quarantine.mask(pool)
    active = unexplained & ~isolated
    return pool[active], active, unexplained, isolated


def _merge_pool(pool, extra):
    return np.unique(np.concatenate((pool, extra)), axis=0)


def termination_coverage(remaining_count, pool_count, config):
    """Use all geometrically unexplained points, including quarantined evidence."""
    rate = remaining_count / pool_count if pool_count else None
    return dict(remaining=remaining_count, total=pool_count, remaining_rate=rate,
                threshold=config.termination_max_remaining_rate,
                passed=bool(pool_count and rate <= config.termination_max_remaining_rate))


def _finish_candidate(query, prefix, pool, rng, validation_rng, config, directory, *,
                      quarantine=None, detection_rng=None, retained_state=None):
    """Confirm coverage, then require all-logit detection, fitting, and independent validation."""
    extra = collect_points_unknown(query, rng, config, 'confirm boundary coverage',
                                   sweeps=config.confirmation_sweeps)
    pool = _merge_pool(pool, extra)
    if retained_state is not None:
        retained_state['pool'] = pool
    remaining, keep = remaining_points(prefix, pool, config, 'confirm unexplained points')
    if quarantine is not None:
        keep &= ~quarantine.discarded_mask(pool)
        remaining = pool[keep]
    np.save(directory / 'coverage_points.npy', pool)
    np.save(directory / 'remaining_mask.npy', keep)
    coverage = termination_coverage(len(remaining), len(pool), config)
    save_json(directory / 'termination_coverage.json', coverage)
    if not coverage['passed']:
        return None, pool, dict(passed=False, reason='confirmation_found_unexplained_points',
                               remaining=len(remaining), coverage=coverage)
    count = max(config.output_samples, 100, 2 * (prefix.structure[-1] + 1))
    inputs = rng.normal(size=(count, query.input_dim))
    detection = []
    output_cache = RecoveryCache()
    with Progress('candidate output rank detection', query.output_dim,
                  config.progress_interval, mode=config.progress_mode) as bar:
        for output in range(query.output_dim):
            result = check_linear_output_system(query.tensor_forward, prefix, inputs,
                                               out_index=output, cache=output_cache)
            detection.append(dict(output=output, consistent=result.consistent,
                                  rank=result.rank, rank_augmented=result.rank_augmented))
            bar.update(output + 1)
    if any(item['consistent'] is not True for item in detection):
        return None, pool, dict(passed=False, reason='output_rank_inconsistent_or_skipped',
                               detection=detection, coverage=coverage)
    inputs = rng.normal(size=(count, query.input_dim))
    inputs += rng.normal(size=inputs.shape) * rng.uniform(-1000, 1000, (count, 1))
    try:
        weight, bias, diagnostics = recover_output_layer(
            query, prefix, inputs, batch_size=config.batch_size, tolerance=config.validation_tolerance,
            progress_mode=config.progress_mode, progress_interval=config.progress_interval)
    except InsufficientEvidence as error:
        return None, pool, dict(passed=False, reason=str(error), detection=detection)
    candidate = copy.deepcopy(prefix)
    candidate.append_layer(weight, bias, layout='in_out')
    candidate.with_relu = False
    error = squared = correct = processed = 0
    with Progress('independent logits validation', config.validation_samples,
                  config.progress_interval, mode=config.progress_mode) as bar:
        for start in range(0, config.validation_samples, config.batch_size):
            size = min(config.batch_size, config.validation_samples - start)
            points = validation_rng.normal(size=(size, query.input_dim))
            target = query_numpy(query, points)
            predicted = candidate.forward_eval(torch.as_tensor(points, device=candidate.device,
                                                                dtype=candidate.dtype)).cpu().numpy()
            difference = predicted - target
            error = max(error, float(np.max(np.abs(difference))))
            squared += float(np.sum(difference ** 2))
            correct += int(np.sum(predicted.argmax(axis=1) == target.argmax(axis=1)))
            processed += size
            bar.update(processed, f'max_abs_error={error:.3g}')
    validation = dict(max_abs_error=error, rmse=float(np.sqrt(squared / (processed * query.output_dim))),
                      label_agreement=correct / processed, samples=processed)
    passed = error <= config.validation_tolerance
    # Released/covered points remain evidence; discarded points are audit-only.
    quarantine_validation = dict(samples=0, max_abs_error=0.0, nonfinite_values=0, passed=True)
    if quarantine is not None and quarantine.entries:
        evidence = quarantine.points(include_discarded=False)
        with Progress('retained quarantine logits validation', len(evidence),
                      config.progress_interval, mode=config.progress_mode) as bar:
            for start in range(0, len(evidence), config.batch_size):
                points = evidence[start:start + config.batch_size]
                target = query_numpy(query, points)
                predicted = candidate.forward_eval(torch.as_tensor(points, device=candidate.device,
                                                                    dtype=candidate.dtype)).cpu().numpy()
                difference = np.abs(predicted - target)
                finite = np.isfinite(difference)
                quarantine_validation['nonfinite_values'] += int((~finite).sum())
                current = float(np.max(difference[finite], initial=0.0))
                quarantine_validation['max_abs_error'] = max(quarantine_validation['max_abs_error'], current)
                quarantine_validation['samples'] += len(points)
                bar.update(start + len(points), f'max_abs_error={quarantine_validation["max_abs_error"]:.3g}')
        quarantine_validation['passed'] = (not quarantine_validation['nonfinite_values']
                                            and quarantine_validation['max_abs_error'] <= config.validation_tolerance)
        passed = passed and quarantine_validation['passed']
    if quarantine is not None and quarantine.summary()['active']:
        review = quarantine.recheck(query, detection_rng, config, stage='before_completion',
                                    passes=config.quarantine_min_confirmations)
        save_json(directory / 'quarantine_review.json', review)
        if review['active']:
            return None, pool, dict(passed=False, reason='unresolved_quarantine', quarantine=review,
                                   remaining=review['active'], coverage=coverage, detection=detection,
                                   validation=validation, quarantine_validation=quarantine_validation)
    if passed:
        np.savez(directory / 'output_layer.npz', weights=weight, biases=bias)
    return (candidate if passed else None), pool, dict(
        passed=passed, reason=('validated' if passed else 'quarantine_output_validation_failed'
                              if not quarantine_validation['passed'] else 'independent_validation_failed'),
        detection=detection, output_layer=diagnostics, validation=validation,
        quarantine_validation=quarantine_validation, remaining=len(remaining), coverage=coverage)


def extract_model_unknown(query, output_dir, config=None, *, initial_points=None):
    """Recover a model using only query dimensions, outputs, and query counts.

    No target structure or parameters are supplied. Rejected candidates are not
    committed; retries add points to the same accepted prefix. Budget exhaustion
    returns incomplete and saves the accepted prefix and rejected candidate.
    """
    from extraction.signature_recovery import PersistentStallTracker, recover_cross_layer
    config = config or UnknownExtractionConfig()
    config.validate_dimensions(query.input_dim, query.output_dim)
    limits = [limit for limit in (query.max_queries, config.max_queries) if limit is not None]
    query.max_queries = min(limits) if limits else None
    pool = (np.empty((0, query.input_dim)) if initial_points is None else
            np.asarray(initial_points, dtype=np.float64).copy())
    if pool.ndim != 2 or pool.shape[1] != query.input_dim or not np.isfinite(pool).all():
        raise ValueError('initial_points must be finite (n, input_dim)')
    pool = np.unique(pool, axis=0)
    output_dir = runtime_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    prefix = RecoveryModel(query.input_dim, device=query.device, dtype=query.dtype, with_relu=True).eval()
    quarantine = PointQuarantine(query.input_dim)
    seeds = np.random.SeedSequence(config.seed).spawn(4)
    rng, detection_rng, validation_rng, refinement_rng = [np.random.default_rng(seed) for seed in seeds]
    progress = dict(progress_mode=config.progress_mode, progress_interval=config.progress_interval)
    report = dict(status='running', information='unknown hidden structure; logits + input/output dimensions',
                  recovered_hidden_layers=0, layers=[], attempts=[], queries=0, deferred_candidates=[],
                  cross_layer_blocks=[])
    save_json(output_dir / 'config.json', dict(input_dim=query.input_dim, output_dim=query.output_dim,
                                             **asdict(config), information=report['information']))
    try:
        if not len(pool):
            pool = collect_points_unknown(query, rng, config, 'initial sampling',
                                          point_count=config.initial_points)
        while True:
            if quarantine.summary()['active']:
                quarantine.recheck(query, detection_rng, config, stage=f'accepted_layer_{len(prefix.fcs)}')
                quarantine.save(output_dir)
            unresolved, _, unexplained, _ = _active_remaining(prefix, pool, quarantine, config, 'accepted prefix coverage')
            if not len(unresolved) or termination_coverage(int(unexplained.sum()), len(pool), config)['passed']:
                # Include zero-hidden-layer models; an empty pool still requires output validation.
                finished = False
                for attempt in range(config.max_rounds):
                    directory = output_dir / f'finish_{len(prefix.fcs):02d}_{attempt + 1}'
                    directory.mkdir()
                    candidate, pool, info = _finish_candidate(
                        query, prefix, pool, rng, validation_rng, config, directory,
                        quarantine=quarantine, detection_rng=detection_rng)
                    save_json(directory / 'report.json', info)
                    if candidate is not None:
                        prefix = candidate
                        report.update(status='complete', stopping_reason='remaining_point_rate_within_threshold_and_output_validated',
                                      validation=info['validation'], output_layer=info['output_layer'],
                                      quarantine_validation=info['quarantine_validation'])
                        torch.save(prefix.state_dict(), output_dir / 'model.pth')
                        finished = True
                        break
                    if info.get('remaining', 0):
                        break
                if finished:
                    break
                if info['reason'] == 'unresolved_quarantine':
                    raise InsufficientEvidence('Unresolved quarantine after final review; accepted prefix needs revision')
                unresolved, _, _, _ = _active_remaining(prefix, pool, quarantine, config,
                                                        'coverage after output rejection')
                if not len(unresolved):
                    raise InsufficientEvidence('Covered point pool but output validation failed; prefix needs revision')
            if len(prefix.fcs) >= config.max_hidden_layers:
                raise InsufficientEvidence('Hidden-layer safety budget reached with unexplained points')
            layer = len(prefix.fcs) + 1
            accepted = False
            incomplete_rounds = 0
            persistent_stall = PersistentStallTracker(config.persistent_match_tolerance)
            evidence_cache = RecoveryCache()
            for attempt in range(config.max_rounds):
                directory = output_dir / f'layer_{layer:02d}' / f'round_{attempt + 1}'
                directory.mkdir(parents=True)
                log_progress(f'Unknown structure: layer={layer} round={attempt + 1}; no expected width')
                if attempt:
                    pool = _merge_pool(pool, collect_points_unknown(query, rng, config, f'layer={layer} resample'))
                unresolved, active, unexplained, isolated = _active_remaining(
                    prefix, pool, quarantine, config, f'layer={layer} input filter')
                np.save(directory / 'input_points.npy', pool)
                np.save(directory / 'recovery_points.npy', unresolved)
                np.save(directory / 'recovery_mask.npy', active)
                np.save(directory / 'input_quarantine_mask.npy', isolated)
                clustering_diagnostics = {}
                try:
                    result = recover_layer_weights_unknown(query, prefix, unresolved, rng,
                        output_index=config.output_index, eps=config.ratio_eps, tolerance=config.cluster_tolerance,
                        min_witnesses=config.min_witnesses, bad_pair_budget=config.bad_pair_budget,
                        cluster_method=config.cluster_method,
                        first_layer_clustering=config.first_layer_clustering,
                        first_layer_direction_tolerance=config.first_layer_direction_tolerance,
                        first_layer_bias_tolerance=config.first_layer_bias_tolerance,
                        clustering_diagnostics=clustering_diagnostics, cache=evidence_cache, **progress)
                finally:
                    save_json(directory / 'first_layer_clustering.json', clustering_diagnostics)
                finite = np.isfinite(result.weights).all(axis=0) & np.isfinite(result.biases)
                incomplete_rounds = incomplete_rounds + 1 if np.any(~finite) else 0
                raw_candidates = len(finite)
                deferred = None
                deferred_source = None
                if (config.defer_incomplete_after_rounds and np.any(~finite) and np.any(finite)
                        and incomplete_rounds >= config.defer_incomplete_after_rounds):
                    # Preserve partial coordinate constraints and all witnesses before selecting a subset.
                    # The pool is deliberately unchanged: candidate deferral never filters evidence.
                    np.savez(directory / 'unselected_signatures.npz', weights=result.weights, biases=result.biases)
                    np.savez(directory / 'unselected_groups.npz',
                             **{f'neuron_{i}': g for i, g in enumerate(result.groups)})
                    selected = np.flatnonzero(finite)
                    deferred_source = result
                    deferred = dict(layer=layer, round=attempt + 1, consecutive_incomplete_rounds=incomplete_rounds,
                        selected_indices=selected.tolist(), deferred_indices=np.flatnonzero(~finite).tolist(),
                        evidence_directory=str(directory.relative_to(output_dir)), accepted_subset=False,
                        interpretation='deferred evidence; not proven absent or unnecessary')
                    report['deferred_candidates'].append(deferred)
                    save_json(directory / 'deferred_candidates.json', deferred)
                    result = replace(result, weights=result.weights[:, selected].copy(),
                        biases=result.biases[selected].copy(), groups=[result.groups[i] for i in selected],
                        candidates=len(selected))
                    log_progress(f'Layer {layer}: testing {len(selected)} finite candidates; '
                                 f'deferred={int((~finite).sum())}; all witness points retained')
                if config.signature_refinement == 'stable' and not len(prefix.fcs) and result.complete:
                    np.savez(directory / 'source_signatures.npz', weights=result.weights, biases=result.biases)
                    np.savez(directory / 'source_groups.npz', **{f'neuron_{i}': group for i, group in enumerate(result.groups)})
                    refinement = {}
                    try:
                        result.weights, result.biases = refine_first_layer_signatures(
                            query, result.weights, result.biases, result.groups, refinement_rng,
                            eps=config.ratio_eps, output_index=config.output_index,
                            direction_factor=config.refinement_direction_factor,
                            max_witnesses=config.refinement_max_witnesses,
                            step_trials=config.refinement_step_trials, tolerance=config.refinement_tolerance,
                            diagnostics=refinement, cache=evidence_cache, **progress)
                    finally:
                        save_json(directory / 'signature_refinement.json', refinement)
                if config.improve_precision:
                    result.weights, result.biases, _ = improve_precision(
                        prefix, result.weights, result.biases, result.groups,
                        tolerance=config.precision_rank_tolerance, rng=rng, cache=evidence_cache, **progress)
                if config.signature_refinement == 'stable' and len(prefix.fcs) and result.complete:
                    np.savez(directory / 'source_signatures.npz', weights=result.weights, biases=result.biases)
                    np.savez(directory / 'source_groups.npz', **{f'neuron_{i}': group for i, group in enumerate(result.groups)})
                    refinement = {}
                    try:
                        result.weights, result.biases = refine_deep_layer_signatures(
                            query, prefix, result.weights, result.biases, result.groups, refinement_rng,
                            eps=config.ratio_eps, output_index=config.output_index,
                            direction_factor=config.refinement_direction_factor,
                            step_trials=config.refinement_step_trials, tolerance=config.refinement_tolerance,
                            diagnostics=refinement, cache=evidence_cache, **progress)
                    finally:
                        save_json(directory / 'signature_refinement.json', refinement)
                width = result.weights.shape[1]
                signs = np.zeros(width, dtype=int)
                info = dict(layer=layer, round=attempt + 1, candidates=width, failed=result.failed,
                            rank_unmerged=result.rank_unmerged, accepted=False, reason='incomplete_candidate_parameters',
                            raw_candidates=raw_candidates, consecutive_incomplete_rounds=incomplete_rounds,
                            deferred_candidates=deferred)
                proposal = None
                pending_quarantine = []
                if result.complete:
                    relative = (prefix.calibrate_boundary_tolerances(
                        result.weights, result.biases, result.groups,
                        multiplier=config.boundary_error_multiplier, maximum=config.boundary_max_relative_tolerance,
                        batch_size=config.batch_size, **progress) if config.boundary_filter == 'adaptive'
                        else np.zeros(width))
                    unsigned = copy.deepcopy(prefix)
                    unsigned.append_layer(result.weights, result.biases, layout='in_out')
                    unsigned.append_boundary_tolerances(relative)
                    save_json(directory / 'boundary_filter.json', dict(
                        mode=config.boundary_filter, tolerance=config.boundary_tolerance,
                        relative_tolerances=unsigned.boundary_filter_profile()))
                    unsigned_remaining, _ = remaining_points(unsigned, pool, config, 'unsigned candidate coverage')
                    if config.sign_method == 'legacy-last-hidden-system' and not len(unsigned_remaining):
                        samples = rng.normal(size=(max(100, 2 * (width + prefix.structure[-1] + 1)), query.input_dim))
                        try:
                            signs = recover_last_hidden_signs(query, prefix, result.weights, result.biases,
                                samples, tolerance=config.validation_tolerance, **progress)
                        except ValueError as error:
                            info['sign_system_error'] = str(error)
                    if not np.all(signs):
                        signs = recover_signs(query, prefix, result.weights, result.groups,
                            min_votes=config.sign_min_votes, confidence=config.sign_confidence,
                            cache=evidence_cache, **progress).signs
                    info['reason'] = 'unresolved_signs'
                    if np.all(signs):
                        proposal = copy.deepcopy(prefix)
                        proposal.append_layer(result.weights * signs, result.biases * signs, layout='in_out')
                        proposal.append_boundary_tolerances(relative)
                        remaining, keep, unexplained, isolated = _active_remaining(
                            proposal, pool, quarantine, config, 'signed candidate coverage')
                        np.save(directory / 'remaining_points.npy', remaining)
                        np.save(directory / 'remaining_mask.npy', keep)
                        np.save(directory / 'unexplained_mask.npy', unexplained)
                        np.save(directory / 'quarantined_mask.npy', isolated)
                        info['remaining'] = len(remaining)
                        info['geometric_remaining'] = int(unexplained.sum())
                        info['quarantine'] = quarantine.summary()
                        coverage = termination_coverage(int(unexplained.sum()), len(pool), config)
                        info['termination_coverage'] = coverage
                        if not coverage['passed']:
                            detection = detect_unknown_prefix(query, proposal, remaining, detection_rng, config)
                            info['detection'] = detection
                            info['accepted'] = detection['passed']
                            info['reason'] = 'detection_passed' if detection['passed'] else 'detection_rejected'
                            if detection['passed'] and (deferred is not None or detection.get('inconsistent', 0)
                                                        or detection.get('skipped', 0)):
                                extra = collect_points_unknown(query, rng, config,
                                    f'layer={layer} independent detection confirmation', sweeps=config.confirmation_sweeps)
                                # A repeated point is not independent confirmation evidence.
                                known = {PointQuarantine.point_id(point) for point in pool}
                                fresh = np.asarray([point for point in np.unique(extra, axis=0)
                                                    if PointQuarantine.point_id(point) not in known],
                                                   dtype=np.float64).reshape(-1, query.input_dim)
                                pool = _merge_pool(pool, extra)
                                confirm_points, _ = remaining_points(proposal, fresh, config,
                                                                     'independent detection coverage')
                                confirmation = detect_unknown_prefix(query, proposal, confirm_points, detection_rng, config)
                                np.save(directory / 'confirmation_points.npy', confirm_points)
                                info['detection_confirmation'] = confirmation
                                info['accepted'] = confirmation['passed']
                                info['reason'] = (('deferred_subset_confirmed' if deferred is not None
                                                   else 'detection_passed_with_quarantine') if confirmation['passed']
                                                  else 'detection_confirmation_rejected')
                                if confirmation['passed']:
                                    pending_quarantine = [(remaining, detection, 'primary_detection'),
                                                          (confirm_points, confirmation, 'independent_confirmation')]
                        else:
                            candidate, pool, finish = _finish_candidate(
                                query, proposal, pool, rng, validation_rng, config, directory,
                                quarantine=quarantine, detection_rng=detection_rng)
                            info['finish'] = finish
                            info['accepted'] = candidate is not None
                            info['reason'] = finish['reason']
                            if candidate is not None:
                                proposal = candidate
                                report.update(status='complete',
                                    stopping_reason='remaining_point_rate_within_threshold_and_output_validated',
                                    validation=finish['validation'], output_layer=finish['output_layer'],
                                    quarantine_validation=finish['quarantine_validation'])
                accepted_through = layer
                stall_detection = info.get('detection')
                finish = info.get('finish', {})
                if (stall_detection is None and finish.get('reason') == 'output_rank_inconsistent_or_skipped'
                        and finish.get('detection')):
                    decisions = [item['consistent'] for item in finish['detection']]
                    # Missing persistent contribution in the last hidden layer:
                    # all-logit output systems replace critical-point decisions.
                    stall_detection = dict(passed=False, sufficient_tests=all(x is not None for x in decisions),
                        inconsistent_rate=sum(x is False for x in decisions) / len(decisions),
                        skipped_rate=sum(x is None for x in decisions) / len(decisions),
                        thresholds=dict(max_inconsistent_rate=0., max_skipped_rate=0.))
                    info['persistent_trigger_test'] = 'all_logit_output_rank'
                stalled = persistent_stall.observe(proposal, stall_detection, len(pool))
                info['persistent_stalled_rounds'] = stalled
                if (not info['accepted'] and config.persistent_after_rounds
                        and stalled >= config.persistent_after_rounds):
                    cross_state = dict(pool=pool)
                    try:
                        cross_proposal, cross_info, cross_quarantine = recover_cross_layer(
                            query, proposal, cross_state, quarantine, rng, detection_rng, validation_rng,
                            config, directory / 'persistent')
                    finally:
                        pool = cross_state['pool']
                    info['persistent'] = cross_info
                    if cross_proposal is not None:
                        proposal, pending_quarantine = cross_proposal, cross_quarantine
                        accepted_through = cross_info['accepted_through']
                        info.update(accepted=True, reason='cross_layer_confirmed')
                        report['cross_layer_blocks'].append({key: cross_info[key] for key in
                            ('layer', 'ordinary_width', 'linear_input_width', 'representation_width',
                             'persistent_count', 'accepted_through')})
                        if cross_info['output_layer']:
                            finish = cross_info['finish']
                            report.update(status='complete',
                                stopping_reason='cross_layer_output_validated', validation=finish['validation'],
                                output_layer=finish['output_layer'], quarantine_validation=finish['quarantine_validation'])
                if info['accepted']:
                    for points, detection, stage in pending_quarantine:
                        quarantine.add(proposal, points, detection, layer=accepted_through,
                                       round_number=attempt + 1, stage=stage)
                    info['quarantine'] = quarantine.summary()
                np.savez(directory / 'signatures.npz', weights=result.weights, biases=result.biases, signs=signs)
                np.savez(directory / 'groups.npz', **{f'neuron_{i}': group for i, group in enumerate(result.groups)})
                if deferred is not None:
                    deferred.update(accepted_subset=bool(info['accepted']), outcome=info['reason'])
                    save_json(directory / 'deferred_candidates.json', deferred)
                info['queries'] = query.query_count
                info['cache'] = evidence_cache.summary()
                save_json(directory / 'report.json', info)
                report['attempts'].append(dict(layer=layer, round=attempt + 1, candidates=width,
                                               accepted=info['accepted'], reason=info['reason']))
                if info['accepted']:
                    prefix = proposal
                    report['recovered_hidden_layers'] = accepted_through
                    report['layers'].extend(dict(layer=number, width=prefix.structure[number])
                                            for number in range(layer, accepted_through + 1))
                    torch.save(prefix.state_dict(), output_dir / ('model.pth' if report['status'] == 'complete' else 'prefix.pth'))
                    save_json(output_dir / 'boundary_filter.json', dict(
                        mode=config.boundary_filter, tolerance=config.boundary_tolerance,
                        relative_tolerances=prefix.boundary_filter_profile()))
                    quarantine.save(output_dir)
                    log_progress(f'Accepted through layer={accepted_through}; saved prefix')
                    accepted = True
                    break
                if proposal is not None:
                    torch.save(proposal.state_dict(), directory / 'rejected_candidate.pth')
                log_progress(f'Retry layer={layer}: {info["reason"]}; candidate not committed')
                if (config.follow_attempts and width and attempt + 1 < config.max_rounds
                        and not prefix.has_linear_coordinates()):
                    follow_result = deferred_source if deferred_source is not None else result
                    extra = get_more_crit_pts(query, prefix, follow_result.weights, follow_result.biases,
                        follow_result.groups, rng, attempts=config.follow_attempts, step=config.follow_step,
                        output_index=config.output_index, grad_eps=config.follow_grad_eps, **progress)
                    pool = _merge_pool(pool, extra)
            if report['status'] == 'complete':
                break
            if not accepted:
                if quarantine.summary()['active']:
                    review = quarantine.recheck(query, detection_rng, config, stage=f'layer_{layer}_stalled',
                                                passes=config.quarantine_min_confirmations)
                    save_json(output_dir / f'quarantine_review_stalled_{layer:02d}.json', review)
                raise InsufficientEvidence(f'Layer {layer} rejected after {config.max_rounds} rounds; unresolved points retained')
    except (InsufficientEvidence, QueryBudgetExceeded) as error:
        report.update(status='incomplete', reason=str(error))
    except BaseException as error:
        report.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                      reason=f'{type(error).__name__}: {error}')
        raise
    finally:
        np.save(output_dir / 'point_pool.npy', pool)
        report.update(queries=query.query_count, saved_structure=list(prefix.structure),
                      partial_with_relu=prefix.with_relu, sampled_points=len(pool), quarantine=quarantine.summary())
        report['model_format'] = ('RecoveryModel with linear_coordinates buffers; widths include linear carriers'
                                  if prefix.has_linear_coordinates() else 'ordinary fully connected ReLU')
        quarantine.save(output_dir)
        torch.save(prefix.state_dict(), output_dir / 'partial.pth')
        save_json(output_dir / 'report.json', report)
        save_json(output_dir / 'deferred_candidates.json', report['deferred_candidates'])
        log_progress(f'Unknown-structure extraction {report["status"]}; queries={query.query_count}; results={output_dir}')
    return report
