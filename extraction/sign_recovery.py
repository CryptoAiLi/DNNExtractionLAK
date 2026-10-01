"""Recover candidate neuron signs using the recovered prefix and output queries."""
from dataclasses import dataclass
import numpy as np
from extraction.utils import query_numpy, prefix_tensor, prefix_values
from utils.progress import Progress


# Use the recovered prefix for local propagation and logits for target queries.
# EPS_IN=1e-6, EPS_LYR=1e-8, EPS_ZERO=1e-12, and SAMPLE_DIFF_ZERO=1e-13.
# Confidence is a vote fraction, not a statistical guarantee.


@dataclass
class SignEvidence:
    signs: np.ndarray
    positive: np.ndarray
    negative: np.ndarray
    skipped: np.ndarray
    confidence: np.ndarray


def recover_signs(query, prefix, weights, groups, *, min_votes=1, confidence=0.5,
                  progress_mode='auto', progress_interval=5.0, cache=None):
    width = weights.shape[1]
    if len(groups) != width or weights.shape[0] != prefix.structure[-1]:
        raise ValueError('Signatures/groups must match the recovered prefix')
    if min_votes <= 0 or not 0.5 <= confidence <= 1:
        raise ValueError('Expected positive min_votes and 0.5 <= confidence <= 1')
    positive = np.zeros(width, dtype=int)
    negative = np.zeros(width, dtype=int)
    skipped = np.zeros(width, dtype=int)
    total = sum(len(group) for group in groups)
    completed = 0
    scope = cache.scope(query, prefix) if cache is not None else None
    with Progress(f'layer={len(prefix.fcs) + 1} signs', total, progress_interval,
                  mode=progress_mode) as progress:
        for neuron, group in enumerate(groups):
            for point in group:
                point = np.asarray(point)
                key = cache.key(scope, 'wiggle_basis', point) if cache is not None else None
                prepared = cache.get(key) if cache is not None else None
                if prepared is None:
                    differences = prefix.local_wiggle_differences(point, eps=1e-6).cpu().numpy()
                    rank = np.linalg.matrix_rank(differences)
                    basis = np.empty((0, weights.shape[0]))
                    if rank:
                        basis = np.linalg.qr(differences.T)[0][:, :rank].T
                        activations = prefix.cheat(prefix_tensor(prefix, point), pad=False)
                        if activations:
                            counts = [int(prefix.activation_mask(layer, prefix_tensor(prefix, a)).sum())
                                      for layer, a in zip(prefix.fcs, activations)]
                            basis = basis[:min(counts)]
                    if cache is not None:
                        cache.put(key, (differences, rank, basis))
                else:
                    differences, rank, basis = prepared
                valid = False
                if rank:
                    projection = np.zeros(weights.shape[0])
                    for vector in basis:
                        projection += (weights[:, neuron] @ vector) / (vector @ vector) * vector
                    projection[np.abs(projection) <= 1e-12] = 0
                    norm = np.linalg.norm(projection)
                    if norm > 0:
                        wiggle = 1e-6 * np.linalg.lstsq(
                            differences.T, projection * (1e-8 / norm), rcond=None)[0]
                        points = np.array([point - wiggle, point + wiggle, point])
                        vote_key = cache.key(scope, 'wiggle_output', points) if cache is not None else None
                        outputs = cache.get(vote_key) if cache is not None else None
                        if outputs is None:
                            outputs = query_numpy(query, points)
                            if cache is not None:
                                cache.put(vote_key, outputs)
                        left, right = np.linalg.norm(outputs[:2] - outputs[2], axis=1)
                        if np.isfinite(left + right) and abs(left - right) >= 1e-13:
                            positive[neuron] += right > left
                            negative[neuron] += left > right
                            valid = True
                skipped[neuron] += not valid
                completed += 1
                progress.update(completed, f'neuron={neuron + 1}/{width} '
                                f'valid={int((positive + negative).sum())} skipped={int(skipped.sum())}')
    votes = positive + negative
    proportions = np.divide(np.maximum(positive, negative), votes,
                            out=np.zeros(width), where=votes > 0)
    signs = np.sign(positive - negative).astype(int)
    signs[(votes < min_votes) | (proportions < confidence)] = 0
    return SignEvidence(signs, positive, negative, skipped, proportions)


def _solve_last_hidden_signs(query, prefix, weights, biases, points, *, tolerance=1e-5):
    """Solve final-hidden-layer signs using f = abs(z) A + h B + c.

    The relation B = W diag(s) A couples signs to a joint fit over all logits.
    """
    hidden = prefix_values(prefix, points)
    z = hidden @ weights + biases
    design = np.column_stack((np.abs(z), hidden, np.ones(len(hidden))))
    outputs = query_numpy(query, points)
    solution, _, rank, _ = np.linalg.lstsq(design, outputs, rcond=None)
    width = weights.shape[1]
    if rank != design.shape[1]:
        raise ValueError('Last-hidden sign system is rank deficient; collect more diverse points')
    absolute_coeff = solution[:width]
    linear_coeff = solution[width:-1]
    equations = (weights[:, :, None] * absolute_coeff[None, :, :]).transpose(0, 2, 1).reshape(-1, width)
    raw, _, sign_rank, _ = np.linalg.lstsq(equations, linear_coeff.reshape(-1), rcond=None)
    signs = np.sign(raw).astype(int)
    error = np.max(np.abs(equations @ signs - linear_coeff.reshape(-1)))
    fit_error = np.max(np.abs(design @ solution - outputs))
    if sign_rank < width or np.any(signs == 0) or max(error, fit_error) > tolerance:
        raise ValueError('Last-hidden sign system is ambiguous or inconsistent')
    return signs


def recover_last_hidden_signs(query, prefix, weights, biases, points, *, tolerance=1e-5,
                              progress_mode='auto', progress_interval=5.0):
    """Build and solve a final-hidden-layer sign system with shared progress reporting."""
    with Progress(f'layer={len(prefix.fcs) + 1} sign system', 1, progress_interval,
                  mode=progress_mode) as progress:
        signs = _solve_last_hidden_signs(query, prefix, weights, biases, points, tolerance=tolerance)
        progress.update(1, f'samples={len(points)} signs={len(signs)}')
    return signs


