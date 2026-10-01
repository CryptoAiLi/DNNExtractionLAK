"""Extract unknown hidden architectures from logits using critical points and validation."""
import argparse
from dataclasses import fields
from datetime import datetime
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from models.factory import create_and_load_model
from utils.structure import parse_structure
from extraction.pipeline import UnknownExtractionConfig, extract_model_unknown
from extraction.precision import validate_refinement_options
from models.factory import create_blackbox_model
from utils.progress import log_progress
from utils.paths import project_path, results_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, help='Pure white-box DNN state_dict .pth (fcs.N.weight/bias)')
    parser.add_argument('--replay-report', type=Path,
                        help='Preview CSV/JSON output from a saved report.json without extraction or target queries')
    parser.add_argument('--structure', type=parse_structure,
                        help='Optional loader-only structure; inferred from checkpoint if omitted; never passed to extraction')
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda, or cuda:N')
    parser.add_argument('--output-dir', type=Path, default=None,
                        help='Relative to project results; absolute paths allowed; must not exist')
    parser.add_argument('--critical-points', type=Path, help='Optional initial (n,input_dim) .npy point pool')
    parser.add_argument('--no-improve-precision', dest='improve_precision', action='store_false', default=True)
    defaults = UnknownExtractionConfig()
    parser.add_argument('--initial-points', '--points-per-round', dest='initial_points',
                        type=int, default=2000,
                        help='Initial sampling target, bounded by max-sweeps (default: 2000); nonempty loaded pools are retained')
    parser.add_argument('--additional-points', type=int, default=None,
                        help='Critical points per supplementary/confirmation sampling (default: initial-points)')
    for field in fields(defaults):
        if field.name in ('improve_precision', 'points_per_round', 'initial_points', 'additional_points'):
            continue
        default = getattr(defaults, field.name)
        options = {}
        if field.name == 'validation_tolerance':
            options['help'] = 'Maximum absolute output-fit residual and final logits error (default: 0.002)'
        if field.name == 'first_layer_clustering':
            options['choices'] = ('geometry', 'source')
            options['help'] = 'geometry (default): split first-layer graph groups by full direction and bias; source: disable this extension'
        if field.name == 'first_layer_direction_tolerance':
            options['help'] = 'Maximum sign-aligned unit-normal L2 distance within a first-layer group (default: 1e-4)'
        if field.name == 'first_layer_bias_tolerance':
            options['help'] = 'Maximum sign-aligned unit-normal bias difference within a first-layer group (default: 1e-5)'
        if field.name == 'min_witnesses':
            options['help'] = 'Minimum witnesses per candidate (default: 3)'
        if field.name == 'progress_mode':
            options['choices'] = ('auto', 'bar', 'log', 'off')
        if field.name == 'boundary_filter':
            options['choices'] = ('strict', 'adaptive')
            options['help'] = 'adaptive: normalized witness-calibrated boundary error; strict: legacy absolute threshold'
        if field.name == 'sign_method':
            options['choices'] = ('neuron-wiggle', 'legacy-last-hidden-system')
            options['help'] = 'neuron-wiggle: local sign voting; legacy-last-hidden-system: joint final-hidden-layer sign equations'
        if field.name == 'cluster_method':
            options['choices'] = ('source', 'basic')
            options['help'] = 'source: sampled-axis graph; basic: all-axis graph; neither uses bias'
        if field.name == 'max_hidden_layers':
            options['help'] = 'Safety cap, not target depth; reaching it with unexplained points returns incomplete'
        if field.name == 'confirmation_sweeps':
            options['help'] = 'Candidate/termination confirmation sweep budget; initial sampling uses max-sweeps'
        if field.name == 'max_sweeps':
            options['help'] = 'Sweep budget per initial or retry sampling stage; may stop before the point target'
        if field.name == 'max_queries':
            options['help'] = 'Optional query limit; omitted means unlimited'
        if field.name in ('bad_pair_budget', 'cluster_tolerance', 'follow_step'):
            options['help'] = 'Compatibility option; source mode uses fixed equations and thresholds'
        if field.name == 'follow_grad_eps':
            options['help'] = 'Explicit noise-check finite difference step; avoids reading unknown target depth'
        if field.name == 'detection_max_inconsistent_rate':
            options['help'] = 'Maximum inconsistent/tested fraction (default: 0.01); 0 requires none'
        if field.name == 'detection_max_skipped_rate':
            options['help'] = 'Maximum skipped/total fraction (default: 0.01); 0 requires none'
        if field.name == 'detection_min_tested':
            options['help'] = 'Minimum non-skipped points in each layer detection/confirmation (default: 100)'
        if field.name == 'termination_max_remaining_rate':
            options['help'] = 'Try last-hidden/output validation when geometrically unexplained points / full pool <= this rate (default: 0.02; 0 requires full coverage)'
        if field.name == 'quarantine_min_confirmations':
            options['help'] = 'Consecutive consistent rechecks using the originating prefix before release (default: 2)'
        if field.name == 'quarantine_max_failures':
            options['help'] = 'Discard a quarantined point after this many consecutive failed/skipped rechecks (default: 3); history is retained'
        if field.name == 'defer_incomplete_after_rounds':
            options['help'] = 'Consecutive incomplete rounds before testing finite candidates with mandatory fresh confirmation (default: 2; 0 disables)'
        if field.name == 'persistent_after_rounds':
            options['help'] = 'Fresh-point rounds with matched signed candidates and persistent detection failures before cross-layer recovery (default: 2; 0 disables)'
        if field.name == 'persistent_max_rounds':
            options['help'] = 'Maximum cross-layer recovery rounds per stalled candidate (default: 3)'
        if field.name == 'persistent_match_tolerance':
            options['help'] = 'Positive-scale-normalized signed hyperplane matching tolerance for stall detection (default: 1e-4)'
        if field.name == 'persistent_constraint_tolerance':
            options['help'] = 'Cross-layer homogeneous constraint rank/support tolerance (default: 1e-5); does not change error detection'
        if field.name == 'persistent_pair_budget':
            options['help'] = 'Maximum pair seeds for cross-region constraint intersections per cross-layer round (default: 512)'
        if field.name == 'persistent_max_points':
            options['help'] = 'Maximum sampled witnesses for each cross-layer constraint fit (default: 64); full pool retained for detection'
        if field.name == 'signature_refinement':
            options['choices'] = ('stable', 'source')
            options['help'] = 'stable: first-layer and deep active-coordinate refinement (default); source: original signatures only'
        if field.name == 'refinement_direction_factor':
            options['help'] = 'Fitting directions = factor*(observable_dimension+2), plus held-out directions (default: 2)'
        if field.name == 'refinement_max_witnesses':
            options['help'] = 'Maximum low-norm witnesses remeasured per first-layer cluster (default: 8)'
        if field.name == 'refinement_step_trials':
            options['help'] = 'Test ratio_eps*10**k, k=0..trials-1; require an agreeing adjacent pair (default: 3)'
        if field.name == 'refinement_tolerance':
            options['help'] = 'Relative orientation/held-out/step agreement and unit-direction consensus tolerance (default: 1e-4)'
        parser.add_argument('--' + field.name.replace('_', '-'),
                            type=int if field.name == 'max_queries' else type(default), default=default, **options)
    args = parser.parse_args(argv)
    if args.replay_report is not None:
        if args.model is not None:
            parser.error('--replay-report cannot be combined with --model')
        return args
    if args.model is None:
        parser.error('--model is required unless --replay-report is supplied')
    if args.additional_points is None:
        args.additional_points = args.initial_points
    for name in ('initial_points', 'additional_points'):
        if getattr(args, name) <= 0:
            parser.error('--' + name.replace('_', '-') + ' must be positive')
    args.points_per_round = args.initial_points
    if args.defer_incomplete_after_rounds < 0:
        parser.error('--defer-incomplete-after-rounds must be nonnegative (0 disables)')
    if args.persistent_after_rounds < 0 or args.persistent_max_rounds <= 0:
        parser.error('--persistent-after-rounds must be nonnegative; --persistent-max-rounds must be positive')
    if not np.isfinite(args.persistent_match_tolerance) or args.persistent_match_tolerance <= 0:
        parser.error('--persistent-match-tolerance must be finite and positive')
    if (not np.isfinite(args.persistent_constraint_tolerance)
            or not 0 < args.persistent_constraint_tolerance < 1 or args.persistent_pair_budget <= 0):
        parser.error('--persistent-constraint-tolerance must be in (0, 1); --persistent-pair-budget must be positive')
    if args.persistent_max_points < max(2, args.min_witnesses):
        parser.error('--persistent-max-points must be >= max(2, min-witnesses)')
    for name in ('detection_max_inconsistent_rate', 'detection_max_skipped_rate',
                 'termination_max_remaining_rate'):
        value = getattr(args, name)
        if not np.isfinite(value) or not 0 <= value < 1:
            parser.error('--' + name.replace('_', '-') + ' must be finite and in [0, 1)')
    for name in ('detection_min_tested', 'quarantine_min_confirmations', 'quarantine_max_failures'):
        if getattr(args, name) <= 0:
            parser.error('--' + name.replace('_', '-') + ' must be positive')
    try:
        validate_refinement_options(args.refinement_direction_factor, args.refinement_max_witnesses,
                                    args.refinement_step_trials, args.refinement_tolerance, args.ratio_eps)
    except ValueError as error:
        parser.error(str(error))
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.replay_report is not None:
        from utils.results import export_reports
        source = project_path(args.replay_report)
        if source.name != 'report.json' or not source.is_file():
            raise ValueError('--replay-report must refer to an existing report.json')
        output = results_path(args.output_dir or ('replay_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
        if output.exists():
            raise FileExistsError(f'Refusing to overwrite existing replay directory: {output}')
        rows = export_reports(source.parent, output, replay=True)
        log_progress(f'Replayed {rows[0]["run"]}; no target queries executed; output={output}')
        return 0
    config = UnknownExtractionConfig(**{field.name: getattr(args, field.name) for field in fields(UnknownExtractionConfig)})
    output = results_path(args.output_dir or ('logits_extraction_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
    args.model = project_path(args.model)
    if args.critical_points is not None:
        args.critical_points = project_path(args.critical_points)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite existing extraction directory: {output}')
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if torch.device(device).type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    # Load the local target without exposing hidden structure to extraction or stopping rules.
    log_progress(f'Loading target: {args.model}; device={device}; extraction sees only input/output dimensions')
    model = create_and_load_model(args.structure, args.model, double_precision=True, device=device).eval()
    config.validate_dimensions(model.input_dim, model.output_dim)
    query = create_blackbox_model(model, model.input_dim, model.output_dim, device=device,
                            batch_size=config.batch_size, max_queries=config.max_queries)
    points = None
    if args.critical_points is not None:
        log_progress(f'Loading critical points: {args.critical_points}')
        points = np.load(args.critical_points, allow_pickle=False)
        log_progress(f'Loaded critical points: shape={points.shape}')
    report = extract_model_unknown(query, output, config, initial_points=points)
    return 0 if report['status'] == 'complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
