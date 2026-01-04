"""Data cleaning utilities for speech error patterns."""

import numpy as np


def clean_substitution_error(error_patterns: list | np.ndarray | None) -> list:
    """Remove substitution_error if any other error pattern exists.

    According to the ontology, substitution_error should only be used when
    there is no other specific error pattern (e.g., gliding_r, fronting) to
    describe the substitution. This function enforces that rule by removing
    substitution_error when other patterns are present.

    Args:
        error_patterns: List of error pattern strings, or None/empty

    Returns:
        Cleaned list of error patterns

    Example:
        >>> clean_substitution_error(['fronting', 'substitution_error'])
        ['fronting']
        >>> clean_substitution_error(['substitution_error'])
        ['substitution_error']
        >>> clean_substitution_error([])
        []
    """
    # Handle None or empty cases
    if not isinstance(error_patterns, (list | np.ndarray)):
        return []

    if len(error_patterns) == 0:
        return []

    # Convert to list if numpy array
    if isinstance(error_patterns, np.ndarray):
        error_patterns = error_patterns.tolist()

    # If only one error pattern, keep it (even if it's substitution_error)
    if len(error_patterns) == 1:
        return error_patterns

    # If multiple patterns exist and one is substitution_error, remove it
    if "substitution_error" in error_patterns:
        return [e for e in error_patterns if e != "substitution_error"]

    return error_patterns
