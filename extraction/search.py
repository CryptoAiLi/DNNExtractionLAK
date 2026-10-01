"""Search for critical points using line sweeps."""
from extraction.utils import query_numpy
import numpy as np
from extraction.utils import GlobalConfig, predict_manual_fast


def sweep_for_critical_points(args, model, std=1, dataset=None):
    """Repeatedly sweep random lines and yield critical points.

    std controls the scan range; args selects the output and dataset behavior.
    """
    print("Sweep for critical points")
    while True:
        print("Start another sweep")
        sweep = do_better_sweep(
            args, model, low=-std * 1e3, high=std * 1e3, dataset=dataset
        )
        print("Total intersections found", len(sweep))

        if args.dataset == "cifar10":
            print("The output_index is", args.output_index)

        for point in sweep:
            yield point


def do_better_sweep(
    args, model, offset=None, direction=None, low=-1e3, high=1e3, dataset=None,
    *, query=None, input_dim=None, skip_linear_tol=None
):
    """Find critical points by recursively splitting a line interval.

    When a region contains one kink, intersect its two affine pieces to locate it.
    query selects the output-only path and input_dim avoids reading target structure.
    Without query, evaluate known weights with NumPy. skip_linear_tol overrides
    the linearity threshold; an explicit input_dim defaults to 1e-8.
    Returns the critical points found between low and high.
    """
    if skip_linear_tol is not None:
        SKIP_LINEAR_TOL = skip_linear_tol
    elif input_dim is not None:
        SKIP_LINEAR_TOL = 1e-8
    elif len(model.structure) - 1 == 3:
        SKIP_LINEAR_TOL = 1e-7
    else:
        SKIP_LINEAR_TOL = 1e-8

    shape = model.structure[0] if input_dim is None else input_dim

    if offset is None:
        offset = np.random.normal(0, 1, size=shape).flatten()
    if direction is None:
        direction = np.random.normal(0, 1, size=shape).flatten()

    if query is None:
        weights, biases = model.get_weights_bias(transposition=True)

    def memo_forward_pass(x, c={}):
        """Cache forward evaluations along the current scan line."""
        if x not in c:
            inputs = (offset + direction * x)[np.newaxis, :]
            c[x] = (predict_manual_fast(inputs, weights, biases) if query is None
                    else query_numpy(query, inputs))
            if args.dataset == "cifar10":
                c[x] = c[x].flatten()

        if hasattr(args, 'output_index') and args.dataset == "cifar10":
            return c[x][args.output_index]
        else:
            return c[x]

    relus = []

    def search(low, high):
        """Locate a single ReLU switch by intersecting two affine pieces.

        Estimate slopes from (low, q1) and (high, q3); split the interval when the
        intersection does not satisfy the linearity checks.
        """
        GlobalConfig.crit_query_count += 1
        mid = (low + high) / 2
        y1 = f_low = memo_forward_pass(low)
        f_mid = memo_forward_pass(mid)
        y2 = f_high = memo_forward_pass(high)

        # The elementwise linearity check also supports vector outputs.
        if np.any(np.abs(f_mid - (f_high + f_low) / 2) < SKIP_LINEAR_TOL * (
            (high - low) ** 0.5
        )):
            return
        elif np.any(high - low < 1e-8):
            return
        else:
            q1 = (low + mid) * 0.5
            q3 = (high + mid) * 0.5

            f_q1 = memo_forward_pass(q1)
            f_q3 = memo_forward_pass(q3)

            m1 = (f_q1 - f_low) / (q1 - low)
            m2 = (f_q3 - f_high) / (q3 - high)

            if not np.all(m1 == m2):
                d = high - low
                alpha = (y2 - y1 - d * m2) / (d * m1 - d * m2)

                x_should_be = low + (y2 - y1 - d * m2) / (m1 - m2)
                height_should_be = y1 + m1 * (y2 - y1 - d * m2) / (m1 - m2)

            if np.all(m1 == m2):
                # Equal slopes require splitting the interval.
                pass
            elif (
                np.all(0.25 + 1e-5 < alpha)
                and np.all(alpha < 0.75 - 1e-5)
                and np.max(x_should_be) - np.min(x_should_be) < 1e-5
            ):
                x_should_be = np.median(x_should_be)
                real_h_at_x = memo_forward_pass(x_should_be)

                if np.all(
                    np.abs(real_h_at_x - height_should_be) < SKIP_LINEAR_TOL * 100
                ):
                    # Check gradients and linearity on both sides of the proposed intersection.
                    eighth_left = x_should_be - 1e-4
                    eighth_right = x_should_be + 1e-4
                    grad_left = (memo_forward_pass(eighth_left) - real_h_at_x) / (
                        eighth_left - x_should_be
                    )
                    grad_right = (memo_forward_pass(eighth_right) - real_h_at_x) / (
                        eighth_right - x_should_be
                    )

                    if np.all(np.abs(grad_left - m1) > SKIP_LINEAR_TOL * 10) or np.all(
                        np.abs(grad_right - m2) > SKIP_LINEAR_TOL * 10
                    ):
                        # Reject an intersection that lies in a nonlinear region.
                        pass
                    else:
                        relus.append(offset + direction * x_should_be)
                        return

        search(low, mid)
        search(mid, high)

    search(low, high)

    return relus


if __name__ == "__main__":
    """Search for critical points

    Load a model, sweep for critical points, and save the results.
    """
    import os
    import sys
    import torch
    from argparse import Namespace
    from pathlib import Path

    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

    from models.base import RecoveryModel

    print("=" * 60)
    print("Critical point search")
    print("=" * 60)

    model_path = project_root / "assets" / "pth_64x5_10" / "model.pth"
    output_dir = project_root / "runtime" / "cifar10_64x5_10" / "critical_points"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_structure = [64, 64, 64, 64, 64, 10]

    args = Namespace(
        dataset='cifar10',
        output_index=0,
        layerID=2,
        real_attack=0,
        seed=42,
        Count=3000,
        debug=False,
    )

    print(f"Model path: {model_path}")
    print(f"Output directory: {output_dir}")
    print(f"Model structure: {model_structure}")
    print(f"Dataset: {args.dataset}, output_index: {args.output_index}")
    print("=" * 60)

    print("\n[1/3] Loading model...")
    model = RecoveryModel(model_structure)
    checkpoint = torch.load(model_path, map_location='cpu')
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('model_state_dict',
                     checkpoint.get('state_dict', checkpoint))
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    model.double().cuda()
    print(f"  Input dimension: {model.structure[0]}")
    print(f"  Hidden layers: {model.structure[1:-1]}")
    print(f"  Output dimension: {model.structure[-1]}")

    print("\n[2/3] Searching for critical points...")
    max_points = 100000
    collected = []

    gen = sweep_for_critical_points(args, model, std=1)
    for i, point in enumerate(gen):
        collected.append(point)
        if (i + 1) % 100 == 0:
            print(f"  Collected {i + 1} critical points...")
        if i + 1 >= max_points:
            break

    print(f"\n  Collected {len(collected)} critical points in total")

    print("\n[3/3] Validating and saving...")
    points_by_layer = {}
    valid = 0
    for pt in collected:
        lid, nid = model.on_which_hidden_layer(pt)
        if lid >= 0:
            valid += 1
            points_by_layer.setdefault(lid, []).append((pt, nid))

    print(f"  Valid: {valid}/{len(collected)} ({valid / len(collected) * 100:.1f}%)")
    for lid in sorted(points_by_layer):
        print(f"    Layer {lid}: {len(points_by_layer[lid])} points")

    pts_arr = np.array(collected, dtype=np.float64)
    all_file = output_dir / f"critical_points_{len(collected)}.npy"
    np.save(all_file, pts_arr)

    for lid in sorted(points_by_layer):
        layer_pts = np.array([p for p, _ in points_by_layer[lid]], dtype=np.float64)
        nids = np.array([n for _, n in points_by_layer[lid]], dtype=np.int32)
        np.save(output_dir / f"layer{lid}_critical_points_{len(layer_pts)}.npy", layer_pts)
        np.save(output_dir / f"layer{lid}_neuron_ids.npy", nids)

    print(f"\n  Saved -> {output_dir}")
    print("=" * 60)
