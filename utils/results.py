"""Export historical extraction reports without rerunning target queries."""
import argparse
import csv
import json
from pathlib import Path

from utils.paths import PROJECT_ROOT, results_path


def saved_ed_query_bounds(directory):
    """Reconstruct saved directional checks, excluding undocumented ED stages."""
    lower = upper = checks = skipped = 0
    for path in sorted(directory.glob('layer_*/round_*/report.json')):
        report = json.loads(path.read_text(encoding='utf-8'))
        for key in ('detection', 'detection_confirmation'):
            detection = report.get(key) or {}
            for point in detection.get('points', []):
                directions = point.get('directions', 0)
                if directions <= 0:
                    continue
                checks += 1
                full = 4 + 8 * (directions - 1)
                upper += full
                if point['status'] == 'skipped':
                    skipped += 1
                    lower += 4
                else:
                    lower += full
    return dict(ed_saved_checks=checks, ed_saved_skipped_checks=skipped,
                ed_saved_queries_lower=lower if checks else None,
                ed_saved_queries_upper=upper if checks else None,
                ed_estimate_scope='Saved directional detection and detection_confirmation only; excludes output-rank systems, quarantine reviews and unsaved checks.',
                ed_time_estimate_note='Unavailable: no ED timer or matching hardware benchmark. Query counts alone do not determine wall time.')


def export_reports(source, destination, *, replay=False):
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    directories = [source] if (source / 'report.json').is_file() else sorted(source.glob('logits_extraction_*'))
    for directory in directories:
        report_file = directory / 'report.json'
        if not report_file.exists():
            continue
        report = json.loads(report_file.read_text(encoding='utf-8'))
        validation = report.get('validation') or {}
        row = dict(
            run=directory.name, status=report['status'],
            output_mode='historical_replay' if replay else 'historical_export',
            recovered_structure='-'.join(map(str, report.get('saved_structure', []))),
            accuracy=validation.get('rmse'), max_abs_error=validation.get('max_abs_error'),
            label_agreement=validation.get('label_agreement'), validation_samples=validation.get('samples'),
            total_time_seconds=None, ed_time_seconds=None, ed_time_share=None,
            total_queries=report.get('queries'), queries_this_run=report.get('queries_this_run'),
            ed_queries=None, ed_query_share=None, edr=None, eds=None,
            **saved_ed_query_bounds(directory),
        )
        rows.append(row)
        run_output = destination / directory.name
        run_output.mkdir(exist_ok=True)
        exported = dict(report)
        exported['measurement_summary'] = row
        exported['source_report'] = str(report_file.resolve())
        exported['replay'] = replay
        exported['measurement_notes'] = {
            'accuracy': 'Saved validation output RMSE, not classification accuracy or parameter error.',
            'missing_metrics': 'Historical reports did not record timing, ED cost, or EDR/EDS ground-truth counts.',
            'total_queries': 'Original report query counter; may include prior work for resumed runs. See queries_this_run when available.',
        }
        (run_output / 'results.json').write_text(json.dumps(exported, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    if not rows:
        raise ValueError(f'No extraction reports found in {source}')
    (destination / 'extraction_measurements.json').write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    with (destination / 'extraction_measurements.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=PROJECT_ROOT / 'runtime')
    parser.add_argument('--output-dir', default='')
    args = parser.parse_args()
    output = results_path(args.output_dir)
    rows = export_reports(args.source, output)
    print(f'Exported {len(rows)} attempts to {output}')


if __name__ == '__main__':
    main()
