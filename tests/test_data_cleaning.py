"""Tests for data cleaning utilities."""

import numpy as np

from speech_model.data_cleaning import clean_substitution_error


def test_clean_substitution_error_single_pattern():
    """When substitution_error is the only pattern, keep it."""
    assert clean_substitution_error(["substitution_error"]) == ["substitution_error"]


def test_clean_substitution_error_with_other_patterns():
    """When other patterns exist, remove substitution_error."""
    result = clean_substitution_error(["fronting", "substitution_error"])
    assert result == ["fronting"]
    assert "substitution_error" not in result


def test_clean_substitution_error_multiple_others():
    """When multiple other patterns exist, remove substitution_error."""
    result = clean_substitution_error(["substitution_error", "gliding_r", "voicing"])
    assert set(result) == {"gliding_r", "voicing"}
    assert "substitution_error" not in result


def test_clean_substitution_error_no_patterns():
    """Empty list should return empty list."""
    assert clean_substitution_error([]) == []


def test_clean_substitution_error_none():
    """None should return empty list."""
    assert clean_substitution_error(None) == []


def test_clean_substitution_error_no_subst():
    """When no substitution_error present, return unchanged."""
    patterns = ["fronting", "voicing"]
    result = clean_substitution_error(patterns)
    assert result == patterns


def test_clean_substitution_error_numpy_array():
    """Should work with numpy arrays."""
    patterns = np.array(["fronting", "substitution_error"])
    result = clean_substitution_error(patterns)
    assert result == ["fronting"]


def test_clean_substitution_error_preserves_order():
    """Should preserve the order of remaining patterns."""
    patterns = ["fronting", "substitution_error", "voicing", "gliding_r"]
    result = clean_substitution_error(patterns)
    assert result == ["fronting", "voicing", "gliding_r"]
