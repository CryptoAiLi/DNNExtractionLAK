"""Parse network structures from command-line arguments."""
import argparse


def parse_structure(value):
    """Parse positive layer widths including input and output dimensions."""
    try:
        structure = [int(part) for part in value.split(',')]
    except (ValueError, AttributeError):
        raise argparse.ArgumentTypeError('structure must be comma-separated positive integers') from None
    if len(structure) < 2 or any(width <= 0 for width in structure):
        raise argparse.ArgumentTypeError('structure needs at least two positive dimensions')
    return structure


