"""Evaluate layer-wise missing-neuron faults using ground-truth prefixes.

The default target is 3072-512-512-512-64-10. Random samples refer to sets
of removed neuron indices for each removal count, not to detection points.
"""
import argparse
import csv
import hashlib
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from numbers import Integral
from typing import Dict, Iterable, List, Optional, Sequence, Union

from models.base import WhiteBoxDNN, RecoveryModel
from models.factory import create_and_load_model
from extraction.error_detection import DEFAULT_DIRECTION_FACTOR, DEFAULT_EPS, DEFAULT_ORIENTATION_TOL, DEFAULT_RANK_TOL, check_points, check_linear_output_systems
from models.factory import create_blackbox_model
from extraction.pipeline import UnknownExtractionConfig, collect_points_unknown
from utils.paths import project_path, results_path
from utils.progress import Progress, log_progress


def _placement(model: WhiteBoxDNN):
    """Return the model's (device, dtype)."""
    if isinstance(model, RecoveryModel):
        return model.device, model.dtype
    reference = next(model.parameters(), None)
    if reference is None:
        raise ValueError('Cannot determine device and dtype for a model without parameters')
    return reference.device, reference.dtype


def _normalize_removed(miss_neuron_ids: Iterable[int]) -> List[int]:
    """Validate and sort neuron indices to remove."""
    removed = []
    for neuron_id in miss_neuron_ids:
        if isinstance(neuron_id, bool) or not isinstance(neuron_id, Integral):
            raise ValueError(f'Neuron indices must be integers: {neuron_id!r}')
        removed.append(int(neuron_id))
    if len(set(removed)) != len(removed):
        raise ValueError(f'Duplicate neuron indices: {removed}')
    return sorted(removed)


def build_prefix_model(
    model: WhiteBoxDNN,
    n_layers: int,
    *,
    remove_neuron_ids: Iterable[int] = (),
    device=None,
    dtype=None,
) -> RecoveryModel:
    """Build a prefix containing the first n_layers Linear layers.

    n_layers ranges from 1 to len(model.fcs). remove_neuron_ids applies only to
    the final prefix layer and reduces its output width; an empty tuple leaves
    the prefix intact. Device and dtype default to the source model.
    Returns a RecoveryModel with with_relu=True.
    """
    if not isinstance(model, WhiteBoxDNN):
        raise TypeError('model must be a models.base.WhiteBoxDNN instance')
    total_layers = len(model.fcs)
    if isinstance(n_layers, bool) or not isinstance(n_layers, Integral) or not 1 <= n_layers <= total_layers:
        raise ValueError(f'n_layers must be in [1, {total_layers}]')
    n_layers = int(n_layers)

    removed = _normalize_removed(remove_neuron_ids)
    layer_width = model.structure[n_layers]
    for neuron_id in removed:
        if not 0 <= neuron_id < layer_width:
            raise ValueError(
                f'Neuron index {neuron_id} is outside layer {n_layers - 1} range [0, {layer_width - 1}]'
            )
    removed_set = set(removed)
    keep = [index for index in range(layer_width) if index not in removed_set]

    source_device, source_dtype = _placement(model)
    weights, biases = model.get_weights_bias(transposition=True)
    prefix = RecoveryModel(
        model.structure[0],
        device=source_device if device is None else device,
        dtype=source_dtype if dtype is None else dtype,
        with_relu=True,
    )
    for layer_index in range(n_layers):
        weight, bias = weights[layer_index], biases[layer_index]
        if layer_index == n_layers - 1 and removed:
            weight, bias = weight[:, keep], bias[keep]
        # get_weights_bias(transposition=True) returns (in_features, out_features) weights.
        prefix.append_layer(weight, bias, layout='in_out')
    prefix.eval()
    return prefix


def truncate_prefix(model: WhiteBoxDNN, n_layers: int, *, device=None, dtype=None) -> RecoveryModel:
    """Build a consistent prefix without removing neurons."""
    return build_prefix_model(model, n_layers, device=device, dtype=dtype)


def load_detection_model(
    model_path: Union[str, Path],
    *,
    device='cuda',
    dtype: torch.dtype = torch.float64,
    model_class=WhiteBoxDNN,
) -> WhiteBoxDNN:
    """Load a ground-truth white-box model from a .pth checkpoint.

    Infer structure from state_dict and use strict=False to allow extra metadata
    such as structure. Convert to the requested device and dtype, then call eval.
    An explicit device=None selects CPU; model_class defaults to WhiteBoxDNN.
    """
    model = create_and_load_model(weights_path=model_path, model_class=model_class,
                                  device='cpu' if device is None else device,
                                  dtype=dtype, strict=False)
    model.eval()
    return model


def _as_points(points) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or 0 in array.shape:
        raise ValueError(f'points must be a 2D (n, dim) array; got shape={array.shape}')
    return array


def _hidden_layers(layers: Optional[Iterable[int]], hidden_layer_count: int) -> List[int]:
    if layers is None:
        return list(range(hidden_layer_count))
    selected = []
    for layer_id in layers:
        if isinstance(layer_id, bool) or not isinstance(layer_id, Integral) or not 0 <= layer_id < hidden_layer_count:
            raise ValueError(f'Layer indices must be integers in [0, {hidden_layer_count - 1}]: {layer_id!r}')
        selected.append(int(layer_id))
    return selected


def _hidden_layer_count(model) -> int:
    hidden = len(model.structure) - 2
    if hidden < 1:
        raise ValueError(f'Cannot build a prefix for a model without hidden layers: structure={model.structure}')
    return hidden


def _check_with_progress(model, prefix, points, *, label, progress_mode, progress_interval, **kwargs):
    """Update progress per point or system with throttled refresh and exception-safe cleanup."""
    with Progress(label, len(points), progress_interval, mode=progress_mode) as bar:
        def update(summary):
            bar.update(summary.processed,
                       f'tested={summary.tested} skipped={summary.skipped} '
                       f'inconsistent={summary.inconsistent}')
        if points.ndim == 3:
            return check_linear_output_systems(
                model, prefix, points, progress=update,
                rank_tol=kwargs['rank_tol'], out_index=kwargs['out_index'])
        return check_points(model, prefix, points, progress=update, **kwargs)


def _layer_candidates(model, points, layer_id, *, linear_rng, linear_system_size,
                      progress_mode, progress_interval):
    if layer_id != _hidden_layer_count(model) - 1:
        return filter_prefix_boundary_points(
            points, truncate_prefix(model, layer_id + 1),
            progress_mode=progress_mode, progress_interval=progress_interval)
    width = model.structure[-2]
    rows = 2 * (width + 1) if linear_system_size is None else linear_system_size
    if isinstance(rows, bool) or not isinstance(rows, Integral) or rows <= width + 1:
        raise ValueError('linear_system_size must be an integer greater than the intact final hidden width + 1')
    # Keep affine sampling independent of detection directions and fault-index sampling.
    generator = np.random.default_rng() if linear_rng is None else linear_rng
    systems = np.empty((len(points), rows, model.structure[0]), dtype=np.float64)
    with Progress(f'sample linear systems layer={layer_id}', len(points),
                  progress_interval, mode=progress_mode) as bar:
        for index in range(len(points)):
            systems[index] = generator.normal(0, 1, (rows, model.structure[0]))
            bar.update(index + 1, f'rows_per_system={rows} sampled={index + 1} skipped=0')
    return systems, 0


def _candidate_metadata(candidates):
    linear = candidates.ndim == 3
    return dict(detection_method='linear_output_rank' if linear else 'second_derivative_rank',
                evaluation_unit='system' if linear else 'point',
                points_per_system=int(candidates.shape[1]) if linear else None,
                sampling_distribution='standard_normal' if linear else None)


def filter_prefix_boundary_points(points, prefix, *, progress_mode='off', progress_interval=5.0):
    """Exclude boundaries in every prefix layer, including the final prefix layer.

    Use the intact ground-truth prefix and abs(preactivation) < 1e-5. No active
    dimension filter is added. Nonfinite inputs remain for check_points to count
    as skipped. The caller must supply critical points; later-layer membership
    is not checked here.
    """
    keep = np.ones(len(points), dtype=bool)
    removed = 0
    with Progress(f'filter prefix_layers={len(prefix.fcs)} points', len(points),
                  progress_interval, mode=progress_mode) as bar:
        for start in range(0, len(points), 1024):
            stop = min(start + 1024, len(points))
            batch = points[start:stop]
            finite = np.isfinite(batch).all(axis=1)
            indices = np.flatnonzero(finite) + start
            if len(indices):
                boundary = prefix.prefix_boundary_mask(points[indices]).cpu().numpy()
                keep[indices[boundary]] = False
                removed += int(boundary.sum())
            bar.update(stop, f'kept={stop - removed} filtered={removed}')
    return points[keep], removed


def compute_specificity(
    model,
    points,
    *,
    layers: Optional[Iterable[int]] = None,
    eps: float = DEFAULT_EPS,
    direction_factor: int = DEFAULT_DIRECTION_FACTOR,
    tolerance: float = DEFAULT_ORIENTATION_TOL,
    rank_tol: Optional[float] = DEFAULT_RANK_TOL,
    rng=None,
    linear_rng=None,
    linear_system_size=None,
    out_index: int = 0,
    verbose: bool = False,
    progress=None,
    progress_mode='off',
    progress_interval=5.0,
) -> Dict[int, Dict]:
    """Measure specificity for each hidden layer using an intact ground-truth prefix.

    Rank mismatches are false positives. layers uses zero-based indices and None
    selects all hidden layers. Detection options pass to check_point.
    linear_rng independently samples final-hidden-layer inputs; linear_system_size
    defaults to 2 * (intact hidden width + 1). There is one system per
    supplied point. progress(layer_id) runs at the start of each layer.
    Returns per-layer totals, skipped counts, false positives, and specificity.
    """
    points = _as_points(points)
    hidden = _hidden_layer_count(model)
    results: Dict[int, Dict] = {}
    selected_layers = _hidden_layers(layers, hidden)
    for job, layer_id in enumerate(selected_layers, 1):
        if progress is not None:
            progress(layer_id)
        prefix = truncate_prefix(model, layer_id + 1)
        layer_points, filtered = _layer_candidates(
            model, points, layer_id, linear_rng=linear_rng, linear_system_size=linear_system_size,
            progress_mode=progress_mode, progress_interval=progress_interval)
        summary = _check_with_progress(
            model, prefix, layer_points,
            label=f'specificity job={job}/{len(selected_layers)} layer={layer_id} '
                  f'{"systems" if layer_points.ndim == 3 else "points"}',
            progress_mode=progress_mode, progress_interval=progress_interval,
            eps=eps, direction_factor=direction_factor, tolerance=tolerance,
            rank_tol=rank_tol, rng=rng, out_index=out_index,
        )
        specificity = 0.0 if summary.specificity is None else summary.specificity
        results[layer_id] = {
            **_candidate_metadata(layer_points),
            'input_points': len(points),
            'filtered_points': filtered,
            'candidate_points': len(layer_points),
            'status': 'ok' if summary.tested else 'no_valid_points',
            'total_points': summary.tested,
            'skipped_points': summary.skipped,
            'error_detection_raw_false': summary.inconsistent,
            'error_detection_raw_specificity': specificity,
        }
        if verbose:
            print(
                f'  Layer {layer_id}: tested {summary.tested} {results[layer_id]["evaluation_unit"]} (skipped {summary.skipped}), '
                f'False = {summary.inconsistent}, Specificity = {specificity:.4f}'
            )
    return results


def compute_recall_and_negative_rate(
    model,
    points,
    *,
    layers: Optional[Iterable[int]] = None,
    miss_neuron_counts: Sequence[int] = (1, 2, 3),
    samples_per_config: int = 10,
    eps: float = DEFAULT_EPS,
    direction_factor: int = DEFAULT_DIRECTION_FACTOR,
    tolerance: float = DEFAULT_ORIENTATION_TOL,
    rank_tol: Optional[float] = DEFAULT_RANK_TOL,
    rng=None,
    linear_rng=None,
    linear_system_size=None,
    sampling_rng=None,
    out_index: int = 0,
    verbose: bool = False,
    progress=None,
    progress_mode='off',
    progress_interval=5.0,
) -> Dict:
    """Inject missing-neuron faults and measure recall and negative rates per layer.

    Filter boundaries with the intact prefix, then reuse retained points across
    removal counts and trials. Any rank mismatch detects a fault. Layers with
    no retained points are excluded from the overall recall denominator.
    layers uses zero-based indices. samples_per_config controls fault trials.
    sampling_rng supplies sample for removed indices; direction rng supplies normal.
    A custom rng may supply both. linear_rng independently samples affine inputs,
    shared by trials within a layer; system size defaults to 2 * (intact
    hidden width + 1). progress receives layer, removal count, and trial indices.
    Returns model metadata, per-layer configurations, and overall statistics.
    """
    points = _as_points(points)
    hidden = _hidden_layer_count(model)
    # Use separate Python and NumPy RNGs for fault indices and query directions.
    # Also accept custom RNGs that provide both sample and normal.
    generator = sampling_rng if sampling_rng is not None else (
        rng if callable(getattr(rng, 'sample', None)) else random
    )
    if isinstance(samples_per_config, bool) or not isinstance(samples_per_config, Integral) or samples_per_config <= 0:
        raise ValueError('samples_per_config must be a positive integer')
    miss_neuron_counts = tuple(miss_neuron_counts)
    if not miss_neuron_counts or any(isinstance(n, bool) or not isinstance(n, Integral) or n <= 0 for n in miss_neuron_counts):
        raise ValueError('miss_neuron_counts must contain positive integers')
    results = {
        'model_structure': list(model.structure),
        'total_dual_points': len(points),
        'samples_per_config': samples_per_config,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'layers': {},
    }

    selected_layers = _hidden_layers(layers, hidden)
    filtered_by_layer = {
        layer_id: _layer_candidates(
            model, points, layer_id, linear_rng=linear_rng, linear_system_size=linear_system_size,
            progress_mode=progress_mode, progress_interval=progress_interval)
        for layer_id in selected_layers
    }
    total_jobs = sum(n < model.structure[l + 1] for l in selected_layers
                     if len(filtered_by_layer[l][0])
                     for n in miss_neuron_counts) * samples_per_config
    job = 0
    if total_jobs == 0 and progress_mode != 'off':
        log_progress('recall: no valid configurations; 0 jobs')
    for layer_id in selected_layers:
        layer_size = model.structure[layer_id + 1]
        layer_points, filtered = filtered_by_layer[layer_id]
        layer_results = {'layer_id': layer_id, 'layer_size': layer_size, 'configs': {},
                         **_candidate_metadata(layer_points),
                         'input_points': len(points), 'filtered_points': filtered,
                         'candidate_points': len(layer_points),
                         'status': 'ok' if len(layer_points) else 'no_valid_points'}
        if not len(layer_points):
            if progress_mode != 'off':
                log_progress(f'recall layer={layer_id}: no valid points after filtering; 0 jobs')
            results['layers'][f'layer_{layer_id}'] = layer_results
            continue
        for miss_count in miss_neuron_counts:
            if miss_count >= layer_size:
                # Skip removal counts that would delete the entire layer.
                if progress_mode != 'off':
                    log_progress(f'recall layer={layer_id} miss={miss_count}: skipped (width={layer_size})')
                continue
            config = {
                'miss_count': miss_count,
                'samples': [],
                'summary': {
                    'total_samples': 0,
                    'successful_detections': 0,
                    'total_false_count': 0,
                    'average_negative_rate': 0.0,
                },
            }
            for sample_id in range(samples_per_config):
                job += 1
                miss_neuron_ids = sorted(generator.sample(range(layer_size), miss_count))
                prefix = build_prefix_model(
                    model, layer_id + 1, remove_neuron_ids=miss_neuron_ids
                )
                summary = _check_with_progress(
                    model, prefix, layer_points,
                    label=f'recall job={job}/{total_jobs} layer={layer_id} miss={miss_count} '
                          f'sample={sample_id + 1}/{samples_per_config} '
                          f'{"systems" if layer_points.ndim == 3 else "points"}',
                    progress_mode=progress_mode, progress_interval=progress_interval,
                    eps=eps, direction_factor=direction_factor, tolerance=tolerance,
                    rank_tol=rank_tol, rng=rng, out_index=out_index,
                )
                total_points = summary.tested
                false_count = summary.inconsistent
                negative_rate = false_count / total_points if total_points > 0 else 0.0
                detected = false_count > 0
                config['samples'].append({
                    'sample_id': sample_id,
                    'miss_neuron_ids': miss_neuron_ids,
                    'total_points': total_points,
                    'skipped_points': summary.skipped,
                    'error_detection_raw': {
                        'false_count': int(false_count),
                        'negative_rate': float(negative_rate),
                        'detected': bool(detected),
                    },
                })
                config['summary']['total_samples'] += 1
                if detected:
                    config['summary']['successful_detections'] += 1
                config['summary']['total_false_count'] += false_count
                if progress is not None:
                    progress(layer_id, miss_count, sample_id, samples_per_config)
                if verbose:
                    print(
                        f'  Layer {layer_id}, missing {miss_count} neurons {miss_neuron_ids}: '
                        f'False = {false_count}/{total_points}, rate = {negative_rate:.4f}'
                    )

            if config['summary']['total_samples'] > 0:
                # Include numerical skips in this denominator but exclude prefix-boundary points.
                config['summary']['average_negative_rate'] = float(
                    config['summary']['total_false_count']
                    / (config['summary']['total_samples'] * len(layer_points))
                )
                config['summary']['recall'] = float(
                    config['summary']['successful_detections']
                    / config['summary']['total_samples']
                )
            layer_results['configs'][f'miss_{miss_count}_neurons'] = config
        results['layers'][f'layer_{layer_id}'] = layer_results

    results['overall_statistics'] = _overall_statistics(results)
    return results


def _overall_statistics(results: Dict) -> Dict:
    """Aggregate actual fault trials using each layer's point or system counts."""
    overall = {
        'total_configs_tested': 0,
        'total_samples_tested': 0,
        'overall_recall_ed_raw': 0.0,
        'overall_avg_negative_rate_ed_raw': 0.0,
        'evaluation_unit': 'point_or_system_by_layer',
    }
    successful = 0
    false_total = 0
    candidate_total = 0
    for layer_data in results['layers'].values():
        for config_data in layer_data['configs'].values():
            overall['total_configs_tested'] += 1
            overall['total_samples_tested'] += config_data['summary']['total_samples']
            candidate_total += config_data['summary']['total_samples'] * layer_data['candidate_points']
            for sample in config_data['samples']:
                if sample['error_detection_raw']['detected']:
                    successful += 1
                false_total += sample['error_detection_raw']['false_count']
    if overall['total_samples_tested'] > 0:
        overall['overall_recall_ed_raw'] = successful / overall['total_samples_tested']
        overall['overall_avg_negative_rate_ed_raw'] = false_total / (
            candidate_total
        )
    return overall


def to_jsonable(value):
    """Recursively convert NumPy scalars and arrays to JSON-compatible Python objects."""
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def save_results(results: Dict, output_file: Union[str, Path]) -> Path:
    """Write statistics to JSON and return the output path."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as handle:
        json.dump(to_jsonable(results), handle, indent=2, ensure_ascii=False)
    return output_file


class QueryMeter:
    """Count target queries only; prefix construction, filtering, and propagation are excluded."""
    def __init__(self, model):
        self.model = model
        self.queries = self.calls = 0

    def __enter__(self):
        self.handle = self.model.register_forward_pre_hook(self._count)
        return self

    def _count(self, module, inputs):
        self.queries += 1 if inputs[0].ndim == 1 else len(inputs[0])
        self.calls += 1

    def snapshot(self):
        device = next(self.model.parameters()).device
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        return self.queries, self.calls, time.perf_counter()

    def elapsed(self, start):
        end = self.snapshot()
        return dict(queries=end[0] - start[0], query_batches=end[1] - start[1],
                    elapsed_seconds=end[2] - start[2])

    def __exit__(self, *exc):
        self.handle.remove()


def evaluate_layer(model, points, layer_id, meter, *, seed, progress_mode, progress_interval):
    """Evaluate faults using shared filtering and detection with both negative-rate denominators."""
    start = meter.snapshot()
    progress = dict(progress_mode=progress_mode, progress_interval=progress_interval)
    candidates, filtered = _layer_candidates(
        model, points, layer_id, linear_rng=np.random.default_rng(seed),
        linear_system_size=None, **progress)
    result = dict(layer_id=layer_id, layer_size=model.structure[layer_id + 1],
                  input_points=len(points), filtered_points=filtered,
                  candidate_points=len(candidates), **_candidate_metadata(candidates),
                  preparation=meter.elapsed(start), configs={})

    def check(removed, label, direction_seed):
        began = meter.snapshot()
        prefix = build_prefix_model(model, layer_id + 1, remove_neuron_ids=removed)
        detection_start = meter.snapshot()
        summary = _check_with_progress(
            model, prefix, candidates, label=label, **progress,
            eps=DEFAULT_EPS, direction_factor=DEFAULT_DIRECTION_FACTOR,
            tolerance=DEFAULT_ORIENTATION_TOL, rank_tol=DEFAULT_RANK_TOL,
            rng=np.random.RandomState(direction_seed), out_index=0)
        detection = meter.elapsed(detection_start)
        return dict(miss_neuron_ids=removed, total_points=summary.tested,
                    skipped_points=summary.skipped, false_count=summary.inconsistent,
                    status='ok' if summary.tested else 'no_valid_points',
                    detection=detection,
                    **meter.elapsed(began))

    baseline = check([], f'layer={layer_id} intact', seed)
    tested = baseline['total_points']
    baseline['specificity'] = 1 - baseline['false_count'] / tested if tested else None
    result['specificity'] = baseline
    sampling_rng = random.Random(seed)
    for miss_count in (1, 2):
        if not len(candidates) or miss_count >= result['layer_size']:
            result['configs'][str(miss_count)] = dict(status='no_valid_configuration', samples=[])
            continue
        began = meter.snapshot()
        samples = []
        for sample_id in range(3):
            removed = sorted(sampling_rng.sample(range(result['layer_size']), miss_count))
            sample = check(removed, f'layer={layer_id} miss={miss_count} sample={sample_id + 1}/3',
                           (seed + miss_count * 3 + sample_id) % (2 ** 32))
            tested = sample['total_points']
            sample.update(sample_id=sample_id, detected=sample['false_count'] > 0,
                          negative_rate=sample['false_count'] / tested if tested else 0.0)
            samples.append(sample)
        result['configs'][str(miss_count)] = dict(
            status='ok' if any(s['total_points'] for s in samples) else 'no_valid_points',
            samples=samples, recall=sum(s['detected'] for s in samples) / len(samples),
            average_negative_rate=sum(s['false_count'] for s in samples) / (len(samples) * len(candidates)),
            **meter.elapsed(began))
    result.update(meter.elapsed(start))
    return result


def measurement_summary(report):
    """Table measurements: pooled trial recall and pooled intact-unit specificity.

    Shares and rates are fractions, times are seconds, queries are input rows.
    Only completed layers contribute ED measurements; incomplete runs are marked
    partial. This ideal-prefix experiment does not measure extraction accuracy.
    """
    checks = []
    trials = []
    intact_tested = intact_false = 0
    for layer in report['layers']:
        baseline = layer['specificity']
        checks.append(baseline)
        intact_tested += baseline['total_points']
        intact_false += baseline['false_count']
        for config in layer['configs'].values():
            samples = config['samples']
            checks.extend(samples)
            trials.extend(samples)
    ed_seconds = sum(check['detection']['elapsed_seconds'] for check in checks)
    ed_queries = sum(check['detection']['queries'] for check in checks)
    total_seconds = report.get('elapsed_seconds')
    total_queries = report.get('queries')
    return dict(
        model='-'.join(map(str, report['model_structure'])),
        accuracy=None,
        accuracy_note='Not measured: ground-truth prefixes, no end-to-end extraction.',
        total_time_seconds=total_seconds, ed_time_seconds=ed_seconds,
        ed_time_share=ed_seconds / total_seconds if total_seconds else None,
        total_queries=total_queries, ed_queries=ed_queries,
        ed_query_share=ed_queries / total_queries if total_queries else None,
        edr=sum(sample['detected'] for sample in trials) / len(trials) if trials else None,
        eds=1 - intact_false / intact_tested if intact_tested else None,
        fault_trials=len(trials), valid_fault_trials=sum(sample['total_points'] > 0 for sample in trials),
        intact_tested_units=intact_tested, intact_false_positives=intact_false,
        completed_layers=len(report['layers']), partial=report['status'] != 'complete',
        units='seconds; input-row queries; shares/EDR/EDS are fractions (multiply by 100 for percent)',
        ed_scope='Detection calls only, excluding prefix construction and candidate preparation; completed layers only.',
        edr_definition='Fault trials with at least one rank mismatch / all completed fault trials (including trials with no valid units).',
        eds_definition='1 - pooled intact rank mismatches / pooled tested intact units; points and systems pooled by layer.',
    )


def layer_measurements(report):
    """Normalize intact-baseline detection cost by retained candidates, including skips.

    For the final hidden layer a candidate is a 130-input system for width 64.
    Cost describes one intact check, rather than summing repeated fault trials.
    """
    rows = []
    for layer in report['layers']:
        baseline = layer['specificity']
        trials = [sample for config in layer['configs'].values() for sample in config['samples']]
        candidates = layer['candidate_points']
        detection = baseline['detection']
        rows.append(dict(
            layer=layer['layer_id'] + 1, width=layer['layer_size'],
            edr=sum(s['detected'] for s in trials) / len(trials) if trials else None,
            eds=baseline['specificity'], evaluation_unit=layer['evaluation_unit'],
            points_per_system=layer.get('points_per_system'), candidate_units=candidates,
            ed_queries_per_1000_points=detection['queries'] * 1000 / candidates if candidates else None,
            ed_time_seconds_per_1000_points=detection['elapsed_seconds'] * 1000 / candidates if candidates else None,
            cost_scope='intact baseline detection only; per 1000 retained points or systems, including skipped units',
        ))
    return rows


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='assets/pth_cifar10_512x3_64_10.pth')
    p.add_argument('--points', help='Existing critical-point .npy file; omit to search using target queries')
    p.add_argument('--point-count', type=int, default=100,
                   help='Search target or loaded-pool sample limit; final-layer system count (default: 100)')
    p.add_argument('--max-sweeps', type=int, default=1000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda')
    p.add_argument('--output-dir', help='Relative to project results; absolute paths allowed')
    p.add_argument('--progress-mode', choices=('auto', 'bar', 'log', 'off'), default='auto')
    p.add_argument('--progress-interval', type=float, default=5.0)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.point_count <= 0 or args.max_sweeps <= 0 or not 0 <= args.seed < 2 ** 32:
        raise ValueError('point-count and max-sweeps must be positive; seed must be in [0, 2**32)')
    if not np.isfinite(args.progress_interval) or args.progress_interval <= 0:
        raise ValueError('progress-interval must be finite and positive')
    def log(message):
        if args.progress_mode != 'off':
            log_progress(message)

    started = time.perf_counter()
    model_path = project_path(args.model)
    log(f'Loading target: {model_path}; device={args.device}; dtype=float64')
    model = load_detection_model(model_path, device=args.device)
    output = results_path(args.output_dir or (
        'ideal_error_detection/' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', model_structure=list(model.structure), layers=[])
    save_results(dict(
        **vars(args), model_path=str(model_path), model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
        structure=list(model.structure), dtype='float64', miss_neuron_counts=[1, 2], samples=3,
        information='true weights for intact earlier layers, evaluated layer and boundary filtering; target logits for detection',
        query_unit='target input rows; also record forward batches; loaded point pool historical cost excluded',
        timing='CUDA synchronized wall time; sample includes prefix construction and detection; layer includes preparation',
        eps=DEFAULT_EPS, rank_tol=DEFAULT_RANK_TOL, tolerance=DEFAULT_ORIENTATION_TOL,
        direction_factor=DEFAULT_DIRECTION_FACTOR, out_index=0,
        filtering='reuse reference filter_prefix_boundary_points and current RecoveryModel defaults'), output / 'config.json')
    with QueryMeter(model) as meter:
        try:
            began = meter.snapshot()
            if args.points:
                pool = _as_points(np.load(project_path(args.points), allow_pickle=False))
                if pool.shape[1] != model.structure[0] or not np.isfinite(pool).all():
                    raise ValueError('Critical points have invalid dimensions or nonfinite values')
                indices = np.random.default_rng(args.seed).choice(
                    len(pool), size=min(args.point_count, len(pool)), replace=False)
                points = pool[indices]
                np.save(output / 'point_indices.npy', indices)
            else:
                config = UnknownExtractionConfig(initial_points=args.point_count, max_sweeps=args.max_sweeps,
                                          progress_mode=args.progress_mode, progress_interval=args.progress_interval)
                points = collect_points_unknown(create_blackbox_model(model), np.random.default_rng(args.seed), config,
                                        'collect critical points', point_count=args.point_count)
            report['sampling'] = dict(points=len(points), requested=args.point_count,
                                      source='loaded' if args.points else 'searched', **meter.elapsed(began))
            np.save(output / 'critical_points.npy', points)
            log(f'Saved {len(points)} critical points; output={output}')
            for layer in range(len(model.structure) - 2):
                log(f'Evaluate layer={layer}; width={model.structure[layer + 1]}')
                result = evaluate_layer(model, points, layer, meter, seed=args.seed,
                                        progress_mode=args.progress_mode, progress_interval=args.progress_interval)
                report['layers'].append(result)
                save_results(result, output / f'layer_{layer}.json')
                save_results(report, output / 'results.json')
                log(f'Saved layer={layer}; queries={result["queries"]}; seconds={result["elapsed_seconds"]:.3f}')
            report['status'] = 'complete'
            with (output / 'summary.csv').open('w', newline='', encoding='utf-8-sig') as handle:
                fields = ['layer_id', 'layer_size', 'evaluation_unit', 'miss_count', 'specificity',
                          'recall', 'average_negative_rate', 'queries', 'query_batches', 'elapsed_seconds', 'status']
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for layer in report['layers']:
                    for miss, data in [('0', layer['specificity']), *layer['configs'].items()]:
                        row = {key: layer[key] for key in ('layer_id', 'layer_size', 'evaluation_unit')}
                        row.update({key: data.get(key) for key in fields if key not in row})
                        row.update(miss_count=int(miss), specificity=layer['specificity']['specificity'])
                        writer.writerow(row)
        except BaseException as error:
            report.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                          reason=f'{type(error).__name__}: {error}')
            raise
        finally:
            meter.snapshot()
            report.update(queries=meter.queries, query_batches=meter.calls,
                          elapsed_seconds=time.perf_counter() - started)
            report['measurement_summary'] = measurement_summary(report)
            report['layer_measurements'] = layer_measurements(report)
            save_results(report, output / 'results.json')
    log(f'Completed: {output}; queries={report["queries"]}; seconds={report["elapsed_seconds"]:.3f}')
    return report


if __name__ == '__main__':
    main()
