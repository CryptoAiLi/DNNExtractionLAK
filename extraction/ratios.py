"""Compute directional signatures and weight ratios at critical points."""
import gc
from extraction.utils import query_numpy
import numpy as np
import torch
from tqdm import tqdm
from extraction.utils import get_second_grad_unsigned, basis, to_tensor, compute_weight_difference, get_prev_ratio, AcceptableFailure, CriticalPoint


def get_ratios(args, model, critical_points, dimInput, with_sign=True, eps=1e-5):
    """Estimate first-layer weight ratios using coordinate-wise directional differences.

    with_sign additionally measures combined directions to recover relative signs.
    Returns one ratio vector per critical point.
    """
    weights, biases = model.get_weights_bias(transposition=True)
    N = [range(dimInput)]
    ratios = []
    for j, point in enumerate(critical_points):
        ratio = []
        for i in N[j]:
            ratio.append(
                get_second_grad_unsigned(
                    args, point, basis(i, dimInput), weights, biases, eps, eps / 3
                )
            )

        # Recover relative signs using combined coordinate directions.
        if with_sign:
            both_ratio = []
            for i in N[j]:
                both_ratio.append(
                    get_second_grad_unsigned(
                        args,
                        point,
                        (basis(i, dimInput) + basis(N[j][0], dimInput)) / 2,
                        weights,
                        biases,
                        eps,
                        eps / 3,
                    )
                )

            signed_ratio = []
            for i in range(len(ratio)):
                positive_error = abs(abs(ratio[0] + ratio[i]) / 2 - abs(both_ratio[i]))
                negative_error = abs(abs(ratio[0] - ratio[i]) / 2 - abs(both_ratio[i]))

                if positive_error > 1e-2 and negative_error > 1e-2:
                    print("[Potential Error] Probably something is borked")
                    print(
                        "d^2(e(i))+d^2(e(j)) != d^2(e(i)+e(j))",
                        positive_error,
                        negative_error,
                    )
                    raise AcceptableFailure()

                if positive_error < negative_error:
                    signed_ratio.append(ratio[i])
                else:
                    signed_ratio.append(-ratio[i])
        else:
            signed_ratio = ratio

        ratio = np.array(signed_ratio)
        ratios.append(ratio)

    return ratios


def get_ratios_lstsq(args, model, critical_points, recovery_model, eps=1e-5,
                     *, rng=None, reference_direction_count=False):
    """Estimate weight ratios from random directions using least squares.

    The recovery model contains the known prefix. rng defaults to np.random.
    reference_direction_count counts nonzero final preactivations without ReLU.
    The real_attack branch applies column normalization.
    Returns (ratios, h_matrices).
    """
    print("EPS", eps)
    weights, biases = model.get_weights_bias(transposition=True)
    ratios = []
    h_matrices = []

    for i, point in enumerate(critical_points):
        d_matrix = []
        ys = []

        # Use N+2 random directions for redundant equations.
        LAYER = len(recovery_model.fcs)
        if reference_direction_count:
            # Only the final prefix layer omits ReLU when counting preactivations.
            upper_bound = recovery_model.forward_eval(
                to_tensor(point), with_relu=False
            ).cpu().numpy()
        else:
            upper_bound = recovery_model.forward_eval(to_tensor(point), n_layers=LAYER).numpy()
        upper_bound = np.sum(upper_bound != 0) + 2
        print("Upper bound for random directions is", upper_bound)

        for i in range(upper_bound):
            d = np.sign((np.random if rng is None else rng).normal(0, 1, point.shape))
            d_matrix.append(d)

            ratio_val = get_second_grad_unsigned(
                args, point, d, weights, biases, eps, eps / 3
            )

            # Align directional measurements to the reference sign.
            if len(ys) > 0:
                both_ratio_val = get_second_grad_unsigned(
                    args, point, (d + d_matrix[0]) / 2, weights, biases, eps, eps / 3
                )

                positive_error = abs(abs(ys[0] + ratio_val) / 2 - abs(both_ratio_val))
                negative_error = abs(abs(ys[0] - ratio_val) / 2 - abs(both_ratio_val))

                if positive_error > 1e-2 and negative_error > 1e-2:
                    print("Probably something is borked")
                    print(
                        "d^2(e(i))+d^2(e(j)) != d^2(e(i)+e(j))",
                        positive_error,
                        negative_error,
                    )
                    raise AcceptableFailure()

                if negative_error < positive_error:
                    ratio_val *= -1

            ys.append(ratio_val)

        d_matrix = np.array(d_matrix)
        h_matrix = recovery_model.forward_at(
            to_tensor(point), torch.from_numpy(d_matrix).double(), n_layers=LAYER
        ).cpu().numpy()

        if len(recovery_model.fcs) != 0:
            # Identify unobservable coordinates.
            column_is_zero = np.mean(np.abs(h_matrix) < 1e-8, axis=0) > 0.5
            relu_out = recovery_model.forward_eval(
                to_tensor(point), n_layers=LAYER, with_relu=True
            ).numpy()
            assert np.all((relu_out == 0) == column_is_zero)

            if hasattr(args, 'real_attack') and args.real_attack == 1:
                # Normalize columns of the local direction matrix.
                normalize_baseline = 0
                abs_sum = []
                for i in range(h_matrix.shape[0]):
                    abs_sum.append(np.sum(np.abs(h_matrix[i])))
                median_value = np.median(abs_sum)
                for i in range(len(abs_sum)):
                    if abs_sum[i] == median_value:
                        normalize_baseline = i
                        break
                print("The normalize baseline is", normalize_baseline)

                h_matrix_non_zero_first_row = []
                h_matrix_non_zero_column_index = []
                for i in range(len(h_matrix[normalize_baseline])):
                    if (
                        abs(h_matrix[normalize_baseline][i]) != 0
                        and h_matrix[normalize_baseline][i] is not None
                    ):
                        h_matrix_non_zero_first_row.append(h_matrix[normalize_baseline][i])
                        h_matrix_non_zero_column_index.append(i)
                print("The non-zero first row of h_matrix is ", h_matrix_non_zero_first_row)
                print("The non-zero column index of h_matrix is ", h_matrix_non_zero_column_index)

                if len(h_matrix_non_zero_first_row) == 1:
                    normalized_h_matrix_non_zero_first_row = np.array(
                        h_matrix_non_zero_first_row
                    )
                else:
                    normalized_h_matrix_non_zero_first_row = (
                        2
                        * (
                            h_matrix_non_zero_first_row
                            - np.min(h_matrix_non_zero_first_row)
                        )
                        / (
                            np.max(h_matrix_non_zero_first_row)
                            - np.min(h_matrix_non_zero_first_row)
                        )
                        - 1
                    )
                print("The normalized non-zero first row is ", normalized_h_matrix_non_zero_first_row)
                normalized_ratio_h_matrix_non_zero_first_row = (
                    normalized_h_matrix_non_zero_first_row / h_matrix_non_zero_first_row
                )
                print("The normalized ratio of the non-zero first row of h_matrix is ",
                      normalized_ratio_h_matrix_non_zero_first_row)

                normalization_ratio = np.zeros(h_matrix.shape[1])
                for i in range(len(normalization_ratio)):
                    if i in h_matrix_non_zero_column_index:
                        normalization_ratio[i] = (
                            normalized_ratio_h_matrix_non_zero_first_row[
                                h_matrix_non_zero_column_index.index(i)
                            ]
                        )

                # Scale columns by their normalization ratios.
                normalized_h_matrix = h_matrix.copy()
                for i in range(normalized_h_matrix.shape[1]):
                    if i in h_matrix_non_zero_column_index:
                        normalized_h_matrix[:, i] = (
                            normalized_h_matrix[:, i]
                            * normalized_ratio_h_matrix_non_zero_first_row[
                                h_matrix_non_zero_column_index.index(i)
                            ]
                        )

                soln_normalized_h_matrix, *rest = np.linalg.lstsq(
                    np.array(normalized_h_matrix, dtype=np.float64),
                    np.array(ys, dtype=np.float64),
                    1e-5,
                )
                soln_normalized_h_matrix[column_is_zero] = np.nan
                soln = soln_normalized_h_matrix

                if hasattr(args, 'debug') and args.debug:
                    print("rank of h_matrix is: ", np.linalg.matrix_rank(h_matrix))
                    print("The non-zero columns are: ", h_matrix.shape[1] - np.sum(column_is_zero))

                # Map the solution back to the unnormalized coordinates.
                for i in range(len(soln)):
                    if i in h_matrix_non_zero_column_index:
                        soln[i] = (
                            soln[i]
                            * normalized_ratio_h_matrix_non_zero_first_row[
                                h_matrix_non_zero_column_index.index(i)
                            ]
                        )
            if hasattr(args, 'real_attack') and args.real_attack == 0:
                # Solve using the unnormalized local matrix.
                soln, *rest = np.linalg.lstsq(
                    np.array(h_matrix, dtype=np.float64),
                    np.array(ys, dtype=np.float64),
                    1e-5,
                )
                for i in range(len(soln)):
                    if column_is_zero[i]:
                        soln[i] = np.nan

        else:
            # Identify unobservable input coordinates.
            if hasattr(args, 'layerID') and args.layerID > 1:
                column_is_zero = np.mean(np.abs(h_matrix) < 1e-8, axis=0) > 0.5
                relu_out = recovery_model.forward_eval(
                    to_tensor(point), n_layers=LAYER, with_relu=True
                ).numpy()
                assert np.all((relu_out == 0) == column_is_zero)

            soln, *rest = np.linalg.lstsq(
                np.array(h_matrix, dtype=np.float64),
                np.array(ys, dtype=np.float64),
                1e-5,
            )

            if hasattr(args, 'debug') and args.debug:
                print("rank of h_matrix is: ", np.linalg.matrix_rank(h_matrix))
                print("The non-zero columns are: ", h_matrix.shape[1] - np.sum(column_is_zero))

        ratios.append(soln)
        h_matrices.append(h_matrix)

    return ratios, h_matrices


def get_ratios_lstsq_with_filter(args, model, critical_points, recovery_model, eps=1e-5):
    """Estimate least-squares ratios while filtering weak directional measurements.

    Returns (ratios, h_matrices, filtered), where filtered indicates rejected points.
    """
    print("EPS", eps)
    weights, biases = model.get_weights_bias(transposition=True)
    ratios = []
    h_matrices = []
    filtered = False

    for p_index, point in enumerate(critical_points):
        d_matrix = []
        ys = []

        if hasattr(args, 'layerID') and args.layerID == 1 and not (hasattr(args, 'real_attack') and args.real_attack == 1):
            upper_bound = len(point)
        else:
            LAYER_F = len(recovery_model.fcs)
            upper_bound = np.sum(
                recovery_model.forward_eval(to_tensor(point), n_layers=LAYER_F).numpy() != 0
            ) + 2
        print("Upper bound for random directions is", upper_bound)

        layer_id, neuron_id = model.on_which_hidden_layer(point)
        print("This point is on layer", layer_id, "and neuron", neuron_id)

        second_derivatives = []
        bad_quality_points_count = 0
        for i in range(upper_bound):
            d = np.sign(np.random.normal(0, 1, point.shape))

            if hasattr(args, 'layerID') and args.layerID == 1 and hasattr(args, 'real_attack') and args.real_attack == 1:
                # Keep random directions correlated with the input in real_attack mode.
                for j in range(len(point)):
                    if point[j] == 0:
                        d[j] = 0
            if hasattr(args, 'layerID') and args.layerID == 1 and not (hasattr(args, 'real_attack') and args.real_attack == 1):
                # Use only coordinate i for first-layer measurements outside real_attack mode.
                d_zeros = np.zeros(point.shape, dtype=np.float64)
                d_zeros[i] = d[i]
                d = d_zeros

            ratio_val = get_second_grad_unsigned(
                args, point, d, weights, biases, eps, eps / 3
            )

            # Reject measurements below the flatness threshold.
            if layer_id == 0:
                output = np.matmul(point, weights[0]) + biases[0]

            second_derivatives.append(abs(ratio_val))

            # Choose the flatness threshold for the dataset.
            if args.dataset == "mnist":
                dimOfLayer = np.array(weights[0]).shape[1]
                if dimOfLayer == 8:
                    flat_threshold = 1e-4
                if dimOfLayer == 16:
                    flat_threshold = 1e-5
                flat_num = 2
            if args.dataset == "cifar10":
                flat_threshold = 1e-5
                flat_num = 1

            if abs(ratio_val) < flat_threshold:
                if layer_id == 0:
                    print("This bad critical point is on the first hidden layer")
                print(
                    "Ratio value is small", ratio_val,
                    "at point ", p_index, "direction", i,
                )
                real_output = np.matmul(point, weights[0]) + biases[0]
                print("The real output is", real_output[neuron_id])

                bad_quality_points_count += 1
                if bad_quality_points_count >= flat_num:
                    print("Too many bad quality directions, stopping the search for this critical point")
                    filtered = True
                    break
            else:
                d_matrix.append(d)

                # Align directional measurements to the reference sign.
                if len(ys) > 0:
                    both_ratio_val = get_second_grad_unsigned(
                        args, point, (d + d_matrix[0]) / 2, weights, biases, eps, eps / 3
                    )

                    positive_error = abs(abs(ys[0] + ratio_val) / 2 - abs(both_ratio_val))
                    negative_error = abs(abs(ys[0] - ratio_val) / 2 - abs(both_ratio_val))
                    if positive_error > 1e-2 and negative_error > 1e-2:
                        print("Probably something is borked")
                        print(
                            "d^2(e(i))+d^2(e(j)) != d^2(e(i)+e(j))",
                            positive_error, negative_error,
                        )
                        raise AcceptableFailure()

                    if negative_error < positive_error:
                        ratio_val *= -1

                ys.append(ratio_val)

        if filtered:
            print("Filtered out the point", p_index, "because the ratio value was too small")
            continue

        d_matrix = np.array(d_matrix)
        LAYER_F = len(recovery_model.fcs)
        h_matrix = recovery_model.forward_at(
            to_tensor(point), torch.from_numpy(d_matrix).double(), n_layers=LAYER_F
        ).cpu().numpy()
        print("Shape of d_matrix:", d_matrix.shape)

        if len(recovery_model.fcs) != 0:
            column_is_zero = np.mean(np.abs(h_matrix) < 1e-8, axis=0) > 0.5
            relu_out = recovery_model.forward_eval(
                to_tensor(point), n_layers=LAYER_F, with_relu=True
            ).numpy()
            assert np.all((relu_out == 0) == column_is_zero)

            normalize_baseline = 0
            abs_sum = []
            for i in range(h_matrix.shape[0]):
                abs_sum.append(np.sum(np.abs(h_matrix[i])))
            median_value = np.median(abs_sum)
            for i in range(len(abs_sum)):
                if abs_sum[i] == median_value:
                    normalize_baseline = i
                    break
            print("The normalize baseline is", normalize_baseline)

            h_matrix_non_zero_first_row = []
            h_matrix_non_zero_column_index = []
            for i in range(len(h_matrix[normalize_baseline])):
                if (
                    abs(h_matrix[normalize_baseline][i]) != 0
                    and h_matrix[normalize_baseline][i] is not None
                ):
                    h_matrix_non_zero_first_row.append(h_matrix[normalize_baseline][i])
                    h_matrix_non_zero_column_index.append(i)

            if len(h_matrix_non_zero_first_row) == 1:
                normalized_h_matrix_non_zero_first_row = np.array(h_matrix_non_zero_first_row)
            else:
                normalized_h_matrix_non_zero_first_row = (
                    2 * (h_matrix_non_zero_first_row - np.min(h_matrix_non_zero_first_row))
                    / (np.max(h_matrix_non_zero_first_row) - np.min(h_matrix_non_zero_first_row))
                    - 1
                )

            normalized_ratio_h_matrix_non_zero_first_row = (
                normalized_h_matrix_non_zero_first_row / h_matrix_non_zero_first_row
            )

            normalization_ratio = np.zeros(h_matrix.shape[1])
            for i in range(len(normalization_ratio)):
                if i in h_matrix_non_zero_column_index:
                    normalization_ratio[i] = normalized_ratio_h_matrix_non_zero_first_row[
                        h_matrix_non_zero_column_index.index(i)
                    ]

            normalized_h_matrix = h_matrix.copy()
            for i in range(normalized_h_matrix.shape[1]):
                if i in h_matrix_non_zero_column_index:
                    normalized_h_matrix[:, i] = (
                        normalized_h_matrix[:, i]
                        * normalized_ratio_h_matrix_non_zero_first_row[
                            h_matrix_non_zero_column_index.index(i)
                        ]
                    )

            soln_normalized_h_matrix, *rest = np.linalg.lstsq(
                np.array(normalized_h_matrix, dtype=np.float64),
                np.array(ys, dtype=np.float64),
                1e-5,
            )
            soln_normalized_h_matrix[column_is_zero] = np.nan
            soln = soln_normalized_h_matrix

            for i in range(len(soln)):
                if i in h_matrix_non_zero_column_index:
                    soln[i] = (
                        soln[i]
                        * normalized_ratio_h_matrix_non_zero_first_row[
                            h_matrix_non_zero_column_index.index(i)
                        ]
                    )
        else:
            if hasattr(args, 'layerID') and args.layerID > 1:
                column_is_zero = np.mean(np.abs(h_matrix) < 1e-8, axis=0) > 0.5
                relu_out = recovery_model.forward_eval(
                    to_tensor(point), n_layers=LAYER_F, with_relu=True
                ).numpy()
                assert np.all((relu_out == 0) == column_is_zero)

            soln, *rest = np.linalg.lstsq(
                np.array(h_matrix, dtype=np.float64),
                np.array(ys, dtype=np.float64),
                1e-5,
            )

        ratios.append(soln)
        h_matrices.append(h_matrix)

    return ratios, h_matrices, filtered


def gather_ratios(
    args,
    this_layer_critical_point_count,
    critical_points_yielder,
    recovery_model,
    cheat_model,
    check_fn,
    LAYER,
    COUNT,
    model,
    dimInput,
    dimOfPrevLayer,
    dimOfLayer,
    special,
    eps=1e-6,
):
    """Gather partial signatures and separate points by their white-box layer assignment.

    Rank-deficient local matrices may produce unreliable signatures. cheat_model
    supplies ground-truth diagnostics, while check_fn filters invalid points.
    Returns (target_points, remaining_points, bad_point_indices).
    """
    if LAYER == 0:
        print("We are on the first layer")

    point_index = this_layer_critical_point_count
    new_this_layer_critical_points = []
    new_remaining_critical_points = []

    critical_point_imprecision = []
    partial_weights_imprecision = []
    bad_critical_point_indices = []

    for index in tqdm(
        range(len(critical_points_yielder)), desc="Analyzing critical points"
    ):
        print("--------------------------------")
        print("Got a new critical point")
        point = critical_points_yielder[index]

        if LAYER > 0:
            # Exclude points near already recovered neuron boundaries.
            if hasattr(args, 'real_attack') and args.real_attack == 0:
                if recovery_model.prefix_boundary_mask(point).item():
                    print("This critical point is caused by a recovered neuron before the target layer")
                    continue

        if LAYER > 0 and recovery_model.filter_prefix_points(
            point, min_active=2, check_boundary=False
        )[1]:
            print("Not enough hidden values are active to get meaningful data")
            continue

        if not check_fn(point):
            print("This critical point is caused by a recovered neuron on the target layer")
            continue

        if len(model.fcs) == 3:
            GRAD_EPS = 1e1
        else:
            GRAD_EPS = 1e-4

        for EPS in [GRAD_EPS, GRAD_EPS / 10, GRAD_EPS / 100]:
            try:
                ratios, h_matrices = get_ratios_lstsq(
                    args, model, [point], recovery_model, eps=EPS
                )

                ratio = ratios[0].flatten()
                h_matrix = h_matrices[0]

                layer_id, neuron_id = model.on_which_hidden_layer(point)
                print("*****************Precision Checking********************")
                weights_all, biases_all = model.get_weights_bias(transposition=True)
                hidden_output = cheat_model.cheat(to_tensor(point), pad=False)
                print(
                    "[Precision Checkpoint - critical point] Point ID: ", index,
                    "Layer ID: ", layer_id,
                    "Neuron ID: ", neuron_id,
                    "Output: ", hidden_output[layer_id][neuron_id],
                )
                print("The point index is", point_index)

                if hidden_output[layer_id][neuron_id] > 1e-6:
                    print("[Precision Checkpoint - critical point] warning!!! the critical point is bad!!")

                if layer_id == args.layerID - 1:
                    print("The critical point is on the target layer")
                    critical_point_imprecision.append(
                        abs(hidden_output[layer_id][neuron_id])
                    )

                    real_weight = []
                    for i in range(len(weights_all[layer_id])):
                        real_weight.append(weights_all[layer_id][i][neuron_id])
                    ratio = np.array(ratio)
                    real_weight = np.array(real_weight)
                    extracted_ratio = ratio.copy()

                    # Correct recovered coordinates using previous-layer scale factors.
                    previous_extracted_sequence, previous_alpha = get_prev_ratio(
                        args, model, biases_all
                    )
                    real_weight = real_weight[previous_extracted_sequence]
                    for weight_index in range(len(extracted_ratio)):
                        extracted_ratio[weight_index] = (
                            extracted_ratio[weight_index] * previous_alpha[weight_index]
                        )

                    column_is_zero = np.mean(np.abs(h_matrix) < 1e-8, axis=0) > 0.5
                    if np.linalg.matrix_rank(h_matrix) >= h_matrix.shape[1] - np.sum(column_is_zero):
                        print("With enough rank")
                        diff = compute_weight_difference(extracted_ratio, real_weight)
                        partial_weights_imprecision.append(diff)
                        print("The difference is", diff)
                break
            except AcceptableFailure:
                print("Try again with smaller eps")
                continue

        try:
            if LAYER == 0:
                new_this_layer_critical_points.append(
                    CriticalPoint(point_index, ratio, point, h_matrix)
                )
                point_index += 1
                continue

            # Separate critical points using their white-box layer assignment.
            layer_id, neuron_id = model.on_which_hidden_layer(point)
            if layer_id != args.layerID - 1:
                new_remaining_critical_points.append(
                    CriticalPoint(index, ratio, point, h_matrix)
                )
            else:
                new_this_layer_critical_points.append(
                    CriticalPoint(point_index, ratio, point, h_matrix)
                )
                point_index += 1
        except:
            continue

    gc.collect()

    print("-----------------------------------------------")
    critical_point_imprecision = [
        value for value in critical_point_imprecision if value != 0
    ]
    if len(critical_point_imprecision) > 0:
        print(
            "[Precision Checkpoint] The largest imprecision of the critical points in the target layer is",
            np.max(critical_point_imprecision),
            "the smallest imprecision is",
            np.min(critical_point_imprecision),
            "and the average imprecision is",
            np.mean(critical_point_imprecision),
        )
    partial_weights_imprecision = [
        value for value in partial_weights_imprecision if value != 0
    ]
    if len(partial_weights_imprecision) != 0:
        print(
            "[Precision Checkpoint] The largest imprecision of the partial weights in the target layer is",
            np.max(partial_weights_imprecision),
            "the smallest imprecision is",
            np.min(partial_weights_imprecision),
            "and the average imprecision is",
            np.mean(partial_weights_imprecision),
        )

    print("len of new_this_layer_critical_points", len(new_this_layer_critical_points))
    print("len of new_remaining_critical_points", len(new_remaining_critical_points))
    print("len of bad_critical_point_indices", len(bad_critical_point_indices))
    return (
        new_this_layer_critical_points,
        new_remaining_critical_points,
        bad_critical_point_indices,
    )


def gather_ratios_without_filter(
    args,
    this_layer_critical_point_count,
    critical_points_yielder,
    recovery_model,
    check_fn,
    LAYER,
    COUNT,
    model,
    eps=1e-6,
):
    """Gather up to COUNT critical-point signatures without target-layer filtering.

    check_fn and prefix checks still reject invalid points.
    Returns a list of CriticalPoint objects.
    """
    point_index = this_layer_critical_point_count
    new_critical_points = []
    print("Gathering", COUNT, "critical points")

    for index, point in enumerate(critical_points_yielder):
        print("--------------------------------")
        print("Got a new critical point")
        if LAYER > 0:
            if hasattr(args, 'real_attack') and args.real_attack == 0:
                if recovery_model.prefix_boundary_mask(point).item():
                    print("This critical point is caused by a recovered neuron before the target layer")
                    continue
            else:
                dead_neurons = []
                if hasattr(args, 'real_attack') and args.real_attack == 1 and hasattr(args, 'layerID') and args.layerID >= 2:
                    import os
                    for i in range(1, args.layerID):
                        if i == 1:
                            recoveryPath = os.path.join(
                                args.output_path, "recovered results",
                                f"Seed{args.seed}-Count500-RealAttack{args.real_attack}",
                            )
                        else:
                            recoveryPath = os.path.join(
                                args.output_path, "recovered results",
                                f"Seed{args.seed}-Count{args.Count}-RealAttack{args.real_attack}",
                            )
                        recovered_parameters = np.load(
                            os.path.join(recoveryPath, f"recovered_{i}.npz")
                        )
                        dead_neuron = []
                        for idx, b in enumerate(recovered_parameters["extracted_bias"]):
                            if abs(b) == 0:
                                dead_neuron.append(idx)
                        dead_neurons.append(dead_neuron)
                    print("The dead neurons from the previous layer are", dead_neurons)

                if recovery_model.prefix_boundary_mask(
                    point, ignored_neurons=dict(enumerate(dead_neurons))
                ).item():
                    print("This critical point is caused by a recovered neuron before the target layer")
                    continue

        if LAYER > 0 and recovery_model.filter_prefix_points(
            point, min_active=2, check_boundary=False
        )[1]:
            print("Not enough hidden values are active to get meaningful data")
            continue

        if not check_fn(point):
            print("This critical point is caused by a recovered neuron on the target layer")
            continue

        if len(model.fcs) == 3:
            GRAD_EPS = 1e1
        else:
            GRAD_EPS = 1e-4

        for EPS in [GRAD_EPS, GRAD_EPS / 10, GRAD_EPS / 100]:
            try:
                ratios, h_matrices = get_ratios_lstsq(
                    args, model, [point], recovery_model, eps=EPS
                )
                ratio = ratios[0].flatten()
                h_matrix = h_matrices[0]
                break
            except AcceptableFailure:
                print("Try again with smaller eps")
                continue
        try:
            new_critical_points.append(
                CriticalPoint(point_index, ratio, point, h_matrix)
            )
            point_index += 1
        except:
            continue

        print("Up to", len(new_critical_points), "of", COUNT)
        if len(new_critical_points) >= COUNT:
            break
    gc.collect()

    return new_critical_points


def second_difference(query, point, direction, eps, output_index):
    """Compute a four-point directional difference with inner step eps / 3."""
    inner = eps / 3
    points = np.asarray([point + (eps - inner) * direction, point + eps * direction,
                         point - (eps - inner) * direction, point - eps * direction])
    values = query_numpy(query, points)[:, output_index]
    return (values[1] - values[0] + values[3] - values[2]) / inner

