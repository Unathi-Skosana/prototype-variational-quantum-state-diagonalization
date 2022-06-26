"""Tests for template."""
from unittest import TestCase

from vqsd.utils.subroutines import _get_zero_counts_at_indices


class TestSubroutines(TestCase):
    """Tests subroutines utility functions."""

    def test_get_zero_counts_at_indices(self):
        """Tests get_zero_counts_at_indices utility."""

        counts_dict = {"00": 10, "01": 20, "10": 30, "11": 400}

        self.assertEqual(
            _get_zero_counts_at_indices(counts_dict, [0]), {"00": 10, "01": 20}
        )

        self.assertEqual(
            _get_zero_counts_at_indices(counts_dict, [1]), {"00": 10, "10": 30}
        )
