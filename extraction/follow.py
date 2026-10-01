"""Follow critical hyperplanes across recovered-prefix regions and collect witnesses."""
import numpy as np
import torch
from models.base import RecoveryModel
from extraction.utils import prefix_tensor, query_numpy
from extraction.search import do_better_sweep as search
from extraction.utils import AcceptableFailure
import gc
import hashlib
from contextvars import ContextVar
from functools import wraps
from argparse import Namespace
from utils.progress import Progress, log_progress


_prefix_cache = ContextVar('follow_prefix_cache', default=None)


def _reuse_prefixes(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        token = _prefix_cache.set(dict(models={}, bytes=0))
        try:
            return function(*args, **kwargs)
        finally:
            _prefix_cache.reset(token)
    return wrapped


def _prefix(A, B, x):
    cache = _prefix_cache.get()
    dimension = np.asarray(x).shape[-1]
    digest = hashlib.sha256(str(dimension).encode())
    for array in (*A, *B):
        array = np.asarray(array)
        digest.update(repr((array.shape, str(array.dtype))).encode())
        digest.update(array.tobytes())
    key = digest.digest()
    if cache is not None and key in cache['models']:
        return cache['models'][key]
    model = RecoveryModel(dimension, device='cpu', dtype=torch.float64, with_relu=True)
    for weight, bias in zip(A, B):
        model.append_layer(weight, bias, layout='in_out')
    needed = sum(p.numel() * p.element_size() for p in model.parameters())
    if cache is not None and cache['bytes'] + needed <= 128 * 1024 * 1024:
        cache['models'][key] = model
        cache['bytes'] += needed
    return model


def forward(x, A, B, with_relu=False):
    x = np.asarray(x)
    model = _prefix(A, B, x)
    with torch.no_grad():
        value = model(prefix_tensor(model, x), with_relu=with_relu).numpy()
    return value[0] if x.ndim == 1 else value


def get_hidden_layers(x, A, B, flat=False):
    model = _prefix(A, B, x)
    values = [v[0] for v in model.cheat(prefix_tensor(model, x), pad=False)]
    return np.concatenate(values) if flat and values else values


def get_hidden_at(A, B, known_A, known_B, layer, x, prior=True):
    values = get_hidden_layers(x, A + [known_A], B + [known_B])
    return tuple(np.concatenate(values if prior else values[-1:]))


def get_polytope_at(A, B, known_A, known_B, x, prior=True):
    return tuple(np.sign(get_hidden_at(A, B, known_A, known_B, 0, x, prior)).astype(int))


def matmul(x, weight, bias):
    # Route even one-layer affine evaluation through the shared model contract.
    return forward(np.asarray(x), [weight], [bias])


def do_better_sweep(args, model, **kwargs):
    return search(args, model, query=model, input_dim=len(kwargs['offset']), **kwargs)


def get_ratios_lstsq(args, model, points, A, B, eps=1e-5):
    from extraction.signature_recovery import estimate_ratio, InsufficientEvidence
    prefix = _prefix(A, B, points[0])
    ratios, matrices = [], []
    for point in points:
        try:
            ratio, matrix, _ = estimate_ratio(model, prefix, point, args.rng,
                output_index=args.output_index, eps=eps, return_details=True)
        except InsufficientEvidence as error:
            raise AcceptableFailure() from error
        ratios.append(ratio)
        matrices.append(matrix)
    return ratios, matrices


def get_ratios(args, model, points, dimInput, eps=1e-5):
    from extraction.signature_recovery import _directional_differences
    ratios = []
    for point in points:
        differences = iter(_directional_differences(model, point, np.eye(dimInput), eps, args.output_index))
        first = next(differences)
        values = [first]
        for _ in range(1, dimInput):
            value, both = next(differences), next(differences)
            positive = abs(abs(first + value) / 2 - abs(both))
            negative = abs(abs(first - value) / 2 - abs(both))
            if min(positive, negative) > 1e-2:
                raise AcceptableFailure()
            values.append(value if positive < negative else -value)
        ratios.append(np.asarray(values))
    return ratios


# Licensed under MIT.
# Use a caller-owned RNG and output queries; grad_eps is independent of target depth.


def binary_search_towards(A, B, known_A, known_B, start_point, initial_signs, go_direction, maxstep=1000000.0):
    """Compute how far we can walk along the hyperplane until it is in a

    different polytope from a prior layer.
    It is okay if it's in a differnt polytope in a *later* layer, because
    it will still have the same angle.
    (but do it analytically by looking at the signs of the first layer)
    this requires no queries and could be done with math but instead
    of thinking I'm just going to run binary search.
    """
    new_A, new_B = (A + [known_A], B + [known_B])
    initial_hidden = np.array(get_hidden_layers(start_point, new_A, new_B, flat=True))
    delta_hidden_np = (np.array(get_hidden_layers(start_point + 1e-06 * go_direction, new_A, new_B, flat=True)) - initial_hidden) * 1000000.0
    can_go_dist_all = initial_hidden / delta_hidden_np
    can_go_dist = -can_go_dist_all[can_go_dist_all < 0]
    if len(can_go_dist) == 0:
        raise AcceptableFailure()
    can_go_dist = np.min(can_go_dist)
    if can_go_dist > 1000000.0:
        raise AcceptableFailure()
    a_bit_further = start_point + (can_go_dist + 0.0001) * go_direction
    return (a_bit_further, can_go_dist)

def find_plane_angle(args, model, A, B, known_A, known_B, dimInput, multiple_intersection_point, sign_at_init, init_step, exponential_base=1.5):
    """Given an input that's at the multiple intersection point, figure out how

    to continue along the path after it bends.
                /       X    : multiple intersection point
       ......../..      ---- : layer N hyperplane
       .      /  .       |   : layer N+1 hyperplane that bends
       .     /   .
    --------X-----------
       .    |    .
       .    |    .
       .....|.....
            |
            |
    We need to make sure to bend, and not turn onto the layer N hyperplane.
    To do this we will draw a box around the X and intersect with the planes
    and determine the four coordinates. Then draw another box twice as big.
    The first layer plane will be the two points at a consistent angle.
    The second layer plane will have an inconsistent angle.
    Choose the inconsistent angle plane, and make sure we move to a new
    polytope and don't just go backwards to where we've already been.
    """
    success = None
    camefrom = None
    prev_iter_intersections = []
    while True:
        x_dir_base = np.sign(args.rng.normal(size=dimInput)) / dimInput ** 0.5
        y_dir_base = np.sign(args.rng.normal(size=dimInput)) / dimInput ** 0.5
        if np.abs(np.dot(x_dir_base, y_dir_base)) <= dimInput % 2 + 1e-08:
            break
    dead_neurons = []
    MAX = 35
    start = [10] if init_step > 10 else []
    for stepsize in start + list(range(init_step, MAX)):
        x_dir = x_dir_base * exponential_base ** (stepsize - 10)
        y_dir = y_dir_base * exponential_base ** (stepsize - 10)
        top = do_better_sweep(args, model, offset=multiple_intersection_point + x_dir, direction=y_dir, low=-1, high=1)
        bot = do_better_sweep(args, model, offset=multiple_intersection_point - x_dir, direction=y_dir, low=-1, high=1)
        left = do_better_sweep(args, model, offset=multiple_intersection_point + y_dir, direction=x_dir, low=-1, high=1)
        right = do_better_sweep(args, model, offset=multiple_intersection_point - y_dir, direction=x_dir, low=-1, high=1)
        intersections = top + bot + left + right
        if len(intersections) == 2 and stepsize >= 10:
            raise AcceptableFailure()
        if len(intersections) == 0 and stepsize > 20:
            raise AcceptableFailure()
        if len(intersections) > 4 and len(prev_iter_intersections) < 2:
            if exponential_base == 1.2:
                return (None, None, 0)
            else:
                return find_plane_angle(args, model, A, B, known_A, known_B, dimInput, multiple_intersection_point, sign_at_init, init_step, exponential_base=1.2)
        if (len(intersections) > 4 or stepsize > 20) and len(prev_iter_intersections) >= 2:
            next_intersections = np.array(prev_iter_intersections[-1])
            intersections = np.array(prev_iter_intersections[-2])
            candidate = []
            for i, a in enumerate(intersections):
                for j, b in enumerate(intersections):
                    if i == j:
                        continue
                    score = np.sum(((a + b) / 2 - multiple_intersection_point) ** 2)
                    a_to_b = b - a
                    a_to_b /= np.sum(a_to_b ** 2) ** 0.5
                    variance = np.std((next_intersections - a) / a_to_b, axis=1)
                    best_variance = np.min(variance)
                    candidate.append((best_variance, i, j))
            if sorted(candidate)[3][0] < 1e-08:
                raise AcceptableFailure()
            err, index_0, index_1 = min(candidate)
            if err / max(candidate)[0] > 1e-05:
                return (None, None, 0)
            prior_layer_near_zero = np.zeros(4, dtype=bool)
            prior_layer_near_zero[index_0] = True
            prior_layer_near_zero[index_1] = True
            should_fail = False
            for critical_point, is_prior_layer_zero in zip(intersections, prior_layer_near_zero):
                new_A, new_B = (A + [known_A], B + [known_B])
                vs = get_hidden_layers(critical_point, new_A, new_B)
                if is_prior_layer_zero:
                    if all([np.min(np.abs(x)) > 1e-05 for x in vs]):
                        should_fail = True
                if any([np.min(np.abs(x)) < 1e-10 for x in vs]):
                    if not is_prior_layer_zero:
                        should_fail = True
            if should_fail:
                return (None, None, 0)
            for critical_point, is_prior_layer_zero in zip(intersections, prior_layer_near_zero):
                sign_at_crit = sign_to_int(get_polytope_at(A, B, known_A, known_B, critical_point))
                if not is_prior_layer_zero:
                    if sign_at_crit != sign_at_init:
                        success = critical_point
                    else:
                        camefrom = critical_point
            if success is None:
                raise AcceptableFailure()
            break
        if len(intersections) == 4:
            prev_iter_intersections.append(intersections)
    gc.collect()
    return (success, camefrom, min(stepsize, MAX - 3))

def sign_to_int(signs):
    """Convert a list to an integer.

    [-1, 1, 1, -1], -> 0b0110 -> 6
    """
    return int(''.join(('0' if x == -1 else '1' for x in signs)), 2)

def follow_hyperplane(args, LAYER, start_point, A, B, known_A, known_B, model, dead_neurons, dimInput, dimOfPrevLayer, dimOfLayer, special, history=[], MAX_POINTS=1000.0, only_need_positive=False, target_neuron=None):
    """This is the ugly algorithm that will let us recover sign for expansive networks.

    Assumes we have extracted up to layer K-1 correctly, and layer K up to sign.
    start_point is a neuron on layer K+1
    known_T is the transformation that computes up to layer K-1, with
    known_A and known_B being the layer K matrix up to sign.
    We're going to come up with a bunch of different inputs,
    each of which has the same critical point held constant at zero.
    """

    def choose_new_direction_from_minimize(previous_axis):
        """Given the current point which is at a critical point of the next

        layer neuron, compute which direction we should travel to continue
        with finding more points on this hyperplane.
        Our goal is going to be to pick a direction that lets us explore
        a new part of the space we haven't seen before.
        """
        if len(history) == 0:
            which_to_change = 0
            new_perp_dir = perp_dir
            new_start_point = start_point
            initial_signs = get_polytope_at(A, B, known_A, known_B, start_point)
            fn = min if initial_signs[0] == 1 else max
        else:
            neuron_values = np.array([x[1] for x in history])
            neuron_positive_count = np.sum(neuron_values > 1e-05, axis=0)
            neuron_negative_count = np.sum(neuron_values < -1e-05, axis=0)
            mean_plus_neuron_value = neuron_positive_count / (neuron_positive_count + neuron_negative_count + 1)
            mean_minus_neuron_value = neuron_negative_count / (neuron_positive_count + neuron_negative_count + 1)
            if only_need_positive:
                neuron_consistency = mean_plus_neuron_value
            else:
                neuron_consistency = mean_plus_neuron_value * mean_minus_neuron_value
            sorted_neuron_consistency = np.argsort(neuron_consistency)
            which_to_change_index = 0
            while sorted_neuron_consistency[which_to_change_index] in dead_neurons:
                which_to_change_index += 1
            which_to_change = sorted_neuron_consistency[which_to_change_index]
            if which_to_change != previous_axis:
                if previous_axis is not None and neuron_consistency[previous_axis] == neuron_consistency[which_to_change]:
                    which_to_change = previous_axis
                    new_start_point = start_point
                    new_perp_dir = perp_dir
                else:
                    valid_axes = np.where(neuron_consistency == neuron_consistency[which_to_change])[0]
                    best = (np.inf, None, None)
                    for _, potential_hidden_vector, potential_point in history[-1:]:
                        for potential_axis in valid_axes:
                            value = potential_hidden_vector[potential_axis]
                            if np.abs(value) < best[0]:
                                best = (np.abs(value), potential_axis, potential_point)
                    _, which_to_change, new_start_point = best
                    new_perp_dir = perp_dir
            else:
                new_start_point = start_point
                new_perp_dir = perp_dir
            fn = min if neuron_positive_count[which_to_change] > neuron_negative_count[which_to_change] else max
            arg_fn = np.argmin if neuron_positive_count[which_to_change] > neuron_negative_count[which_to_change] else np.argmax
        val = matmul(forward(new_start_point, A, B, with_relu=True), known_A, known_B)[which_to_change]
        initial_signs = get_polytope_at(A, B, known_A, known_B, new_start_point)
        choices = []
        for _ in range(1000):
            random_dir = args.rng.normal(size=dimInput)
            perp_component = np.dot(random_dir, new_perp_dir) / np.dot(new_perp_dir, new_perp_dir) * new_perp_dir
            parallel_dir = random_dir - perp_component
            go_direction = parallel_dir / np.sum(parallel_dir ** 2) ** 0.5
            try:
                a_bit_further, high = binary_search_towards(A, B, known_A, known_B, new_start_point, initial_signs, go_direction)
            except AcceptableFailure:
                continue
            if a_bit_further is None:
                continue
            val = matmul(forward(a_bit_further[np.newaxis, :], A, B, with_relu=True), known_A, known_B)[0][which_to_change]
            choices.append([val, a_bit_further])
        if len(choices) == 0:
            raise AcceptableFailure()
        best_value, multiple_intersection_point = fn(choices, key=lambda x: x[0])
        return (new_start_point, multiple_intersection_point, which_to_change)

    def check_relevant_crits(points_on_plane, neuron_values, neuron_positive_count):
        relevant_points_on_plane = []
        for i in range(len(points_on_plane)):
            relevant_neurons = neuron_positive_count <= 15
            points_contributing = (neuron_values[i] > 1e-05) & relevant_neurons
            if np.any(points_contributing):
                relevant_points_on_plane.append(points_on_plane[i])
            else:
                mask = neuron_values[i] > 1e-05
                neuron_positive_count[mask] -= 1
        return relevant_points_on_plane

    def is_on_prior_layer(query):
        for layer in get_hidden_layers(query, A, B):
            for i in range(len(layer)):
                if layer[i] != 0 and np.abs(layer[i]) < 1e-05:
                    return True
        next_A, next_B = (A + [known_A], B + [known_B])
        next_hidden = forward(query, next_A, next_B)
        for i in range(len(next_hidden)):
            if next_hidden[i] != 0 and np.abs(next_hidden[i]) < 1e-06:
                return True
        return False
    start_box_step = 0
    points_on_plane = []
    current_change_axis = 0
    if target_neuron != None:
        target_neuron_count = 0
        init_target_neuron_count = 0
    else:
        target_neuron_count = 1
        init_target_neuron_count = 0
    iteration = 0
    while True:
        if iteration >= args.max_steps:
            return (points_on_plane, False)
        iteration += 1
        args.update(iteration, len(points_on_plane))
        gc.collect()
        which_polytope = get_polytope_at(A, B, known_A, known_B, start_point, False)
        hidden_vector = get_hidden_at(A, B, known_A, known_B, LAYER, start_point, False)
        sign_at_init = sign_to_int(which_polytope)
        neuron_values = np.array([x[1] for x in history])
        neuron_positive_count = np.sum(neuron_values > 1e-05, axis=0)
        neuron_negative_count = np.sum(neuron_values < -1e-05, axis=0)
        if target_neuron != None:
            target_neuron_count = min(neuron_positive_count[target_neuron], neuron_negative_count[target_neuron])
            if len(points_on_plane) == 0:
                init_target_neuron_count = target_neuron_count
        if len(points_on_plane) > MAX_POINTS:
            return (points_on_plane, False)
        if (np.all(neuron_positive_count > 0) and np.all(neuron_negative_count > 0) or (only_need_positive and np.all(neuron_positive_count > 0))) and target_neuron_count - init_target_neuron_count > 0:
            neuron_values = np.array([get_hidden_at(A, B, known_A, known_B, LAYER, x, False) for x in points_on_plane])
            neuron_positive_count = np.sum(neuron_values > 1e-05, axis=0)
            neuron_negative_count = np.sum(neuron_values < -1e-05, axis=0)
            relevant_points_on_plane = check_relevant_crits(points_on_plane, neuron_values, neuron_positive_count)
            return (relevant_points_on_plane, True)
        try:
            perp_dir, _ = get_ratios_lstsq(args, model, [start_point], A=[], B=[], eps=1e-05)
            perp_dir = perp_dir[0].flatten()
        except AcceptableFailure:
            return (points_on_plane, False)
        try:
            start_point, multiple_intersection_point, new_change_axis = choose_new_direction_from_minimize(current_change_axis)
        except AcceptableFailure:
            return (points_on_plane, False)
        if new_change_axis != current_change_axis:
            try:
                start_point, multiple_intersection_point, current_change_axis = choose_new_direction_from_minimize(None)
            except AcceptableFailure:
                return (points_on_plane, False)
        towards_multiple_direction = multiple_intersection_point - start_point
        step_distance = np.sum(towards_multiple_direction ** 2) ** 0.5
        if step_distance > 1000000.0:
            continue
        if step_distance > 1 or True:
            mid_point = 0.0001 * towards_multiple_direction / np.sum(towards_multiple_direction ** 2) ** 0.5 + start_point
            for range_bound in [0.001, 0.01]:
                mid_points = do_better_sweep(args, model, offset=mid_point, direction=perp_dir / np.sum(perp_dir ** 2) ** 0.5, low=-range_bound, high=range_bound)
                good_mid_points = []
                for point in mid_points:
                    if not is_on_prior_layer(point):
                        good_mid_points.append(point)
                mid_points = good_mid_points
                if len(mid_points) > 0:
                    break
            if len(mid_points) > 0:
                mid_point = mid_points[np.argmin(np.sum((mid_point - mid_points) ** 2, axis=1))]
                towards_multiple_direction = mid_point - start_point
                towards_multiple_direction = towards_multiple_direction / np.sum(towards_multiple_direction ** 2) ** 0.5
                initial_signs = get_polytope_at(A, B, known_A, known_B, start_point)
                try:
                    _, high = binary_search_towards(A, B, known_A, known_B, start_point, initial_signs, towards_multiple_direction)
                except AcceptableFailure:
                    return (points_on_plane, False)
                multiple_intersection_point = towards_multiple_direction * high + start_point
        success = None
        while success is None:
            if start_box_step < 0:
                start_box_step = 0
                which_point = args.rng.integers(0, len(history))
                start_point = history[which_point][2]
                current_change_axis = args.rng.integers(0, dimOfLayer)
                break
            try:
                success, camefrom, stepsize = find_plane_angle(args, model, A, B, known_A, known_B, dimInput, multiple_intersection_point, sign_at_init, start_box_step)
            except AcceptableFailure:
                start_box_step = -10
            start_box_step -= 2
        if success is None:
            continue
        val = matmul(forward(multiple_intersection_point, A, B, with_relu=True), known_A, known_B)[new_change_axis]
        val = matmul(forward(success, A, B, with_relu=True), known_A, known_B)[new_change_axis]
        if stepsize < 10:
            new_move_direction = success - multiple_intersection_point
            new_point_close_to_success = success + new_move_direction * 0.01
            refine = True
            try:
                new_perp_dir, _ = get_ratios_lstsq(args, model, [new_point_close_to_success], A=[], B=[], eps=1e-05)
                new_perp_dir = new_perp_dir[0].flatten()
            except AcceptableFailure:
                refine = False
            if refine:
                for range_bound in [0.0001, 0.001, 0.01]:
                    new_points = do_better_sweep(args, model, offset=new_point_close_to_success, direction=new_perp_dir / np.sum(new_perp_dir ** 2) ** 0.5, low=-range_bound, high=range_bound)
                    good_new_points = []
                    for point in new_points:
                        if not is_on_prior_layer(point):
                            good_new_points.append(point)
                    new_points = good_new_points
                    if len(new_points) > 0:
                        break
                if len(new_points) > 0:
                    new_point = new_points[np.argmin(np.sum((new_point_close_to_success - new_points) ** 2, axis=1))]
                    new_move_direction = new_point - success
            initial_signs = get_polytope_at(A, B, known_A, known_B, success)
            low = 0
            high = 1
            while high - low > 0.01:
                mid = (high + low) / 2
                query_point = success + mid * new_move_direction
                next_signs = get_polytope_at(A, B, known_A, known_B, query_point)
                if initial_signs == next_signs:
                    low = mid
                else:
                    high = mid
            success = success + mid / 2 * new_move_direction
            val = matmul(forward(success, A, B, with_relu=True), known_A, known_B)[new_change_axis]
        if is_on_target_layer(args, model, A, B, known_A, known_B, success, dimInput):
            start_point = success
            start_box_step = max(stepsize - 1, 0)
            points_on_plane.append(start_point)
            which_polytope = get_polytope_at(A, B, known_A, known_B, start_point, False)
            hidden_vector = get_hidden_at(A, B, known_A, known_B, LAYER, start_point, False)
            history.append((which_polytope, hidden_vector, np.copy(start_point)))
            if camefrom is not None:
                if is_on_target_layer(args, model, A, B, known_A, known_B, camefrom, dimInput):
                    points_on_plane.append(camefrom)
        else:
            return (points_on_plane, False)

def is_on_target_layer(args, model, A, B, known_A, known_B, point, dimInput):
    GRAD_EPS = args.follow_grad_eps

    def is_on_prior_layer(query):
        if any((np.min(np.abs(layer)) < 1e-05 for layer in get_hidden_layers(query, A, B))):
            return True
        next_A, next_B = (A + [known_A], B + [known_B])
        next_hidden = forward(query, next_A, next_B)
        if np.min(np.abs(next_hidden)) < 1e-06:
            return True
        return False
    if is_on_prior_layer(point):
        return False
    initial_signs = get_polytope_at(A, B, known_A, known_B, point)
    try:
        normal = get_ratios(args, model, [point], dimInput, eps=GRAD_EPS)[0].flatten()
    except AcceptableFailure:
        return False
    normal = normal / np.sum(normal ** 2) ** 0.5
    for tol in range(100):
        random_dir = args.rng.normal(size=dimInput)
        perp_component = np.dot(random_dir, normal) / np.dot(normal, normal) * normal
        parallel_dir = random_dir - perp_component
        go_direction = parallel_dir / np.sum(parallel_dir ** 2) ** 0.5
        try:
            _, distance = binary_search_towards(A, B, known_A, known_B, point, initial_signs, go_direction)
        except AcceptableFailure:
            continue
        high_bound = 0.001
        low_bound = -0.001
        point_in_same_polytope = point + (distance * 0.99 - 0.0001) * go_direction
        f_low = query_numpy(model, (point_in_same_polytope + normal * low_bound)[np.newaxis, :])
        f_high = query_numpy(model, (point_in_same_polytope + normal * high_bound)[np.newaxis, :])
        f_mid = query_numpy(model, point_in_same_polytope[np.newaxis, :])
        if args.dataset == 'cifar10':
            f_low = f_low[0][args.output_index]
            f_high = f_high[0][args.output_index]
            f_mid = f_mid[0][args.output_index]
        if np.abs(f_mid - (f_high + f_low) / 2) < 1e-08:
            return False
        high_bound = 0.0001
        low_bound = -0.0001
        point_in_different_polytope = point + (distance * 1.1 + 0.1) * go_direction
        f_low = query_numpy(model, (point_in_different_polytope + normal * low_bound)[np.newaxis, :])
        f_high = query_numpy(model, (point_in_different_polytope + normal * high_bound)[np.newaxis, :])
        f_mid = query_numpy(model, point_in_different_polytope[np.newaxis, :])
        if args.dataset == 'cifar10':
            f_low = f_low[0][args.output_index]
            f_high = f_high[0][args.output_index]
            f_mid = f_mid[0][args.output_index]
        if np.abs(f_mid - (f_high + f_low) / 2) > 0.1:
            return False
    return True


@_reuse_prefixes
def get_more_crit_pts(query, prefix, weights, biases, groups, rng, *, attempts=10,
                     step=0.1, output_index=0, progress_mode='auto', progress_interval=5.0,
                     grad_eps=1e-4):
    """Continue a hyperplane using adaptive intersection distances and coverage checks.

    attempts bounds cross-region iterations per component. step is accepted for
    compatibility; intersection distances determine the actual steps. An empty
    prefix has no earlier regions to cross, so the pipeline continues line sampling.
    grad_eps controls finite differences without using the target depth.
    """
    if attempts < 0 or step <= 0 or not np.isfinite(step):
        raise ValueError('Expected nonnegative attempts and finite positive step')
    if not len(prefix.fcs) or attempts == 0:
        if progress_mode != 'off':
            log_progress('hyperplane follow: no prior layer to cross; continue sweep sampling')
        return np.empty((0, prefix.structure[0]))
    recovered_A, recovered_B = prefix.get_weights_bias(transposition=True)
    A, B = recovered_A[:-1], recovered_B[:-1]
    known_A, known_B = recovered_A[-1], recovered_B[-1]
    tasks = [(i, group) for i, group in enumerate(groups) if len(group)]
    found = []
    with Progress(f'layer={len(prefix.fcs) + 1} hyperplane follow', len(tasks) * attempts,
                  progress_interval, mode=progress_mode) as progress:
        for task, (neuron, group) in enumerate(tasks):
            history = [(get_polytope_at(A, B, known_A, known_B, point, False),
                        get_hidden_at(A, B, known_A, known_B, 0, point, False),
                        np.array(point, copy=True)) for point in group]

            def update(iteration, count):
                progress.update(task * attempts + iteration,
                    f'neuron={neuron + 1} found={len(found) + count}')

            args = Namespace(dataset='cifar10', layerID=len(prefix.fcs) + 1,
                             output_index=output_index, rng=rng, follow_grad_eps=grad_eps,
                             max_steps=attempts, update=update)
            dead = np.flatnonzero(np.all(known_A == 0, axis=0)).tolist()
            if len(dead) == known_A.shape[1]:
                progress.update((task + 1) * attempts, f'neuron={neuron + 1} skipped=dead prefix')
                continue
            try:
                extra, _ = follow_hyperplane(args, len(A), np.array(group[0]), A, B,
                    known_A, known_B, query, dead, prefix.structure[0], known_A.shape[0],
                    known_A.shape[1], None, history=history, MAX_POINTS=200,
                    only_need_positive=True)
                found.extend(extra)
            except AcceptableFailure:
                pass
            progress.update((task + 1) * attempts, f'neuron={neuron + 1} found={len(found)}')
    return np.asarray(found, dtype=np.float64).reshape(-1, prefix.structure[0])
