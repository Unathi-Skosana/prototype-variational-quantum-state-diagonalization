"""Tests for template."""
from unittest import TestCase

import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit.library import EfficientSU2

from vqsd.utils.subroutines import (
    _get_zero_counts_at_indices,
    prepare_circuits_to_execute,
)


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

    def test_prepare_circuits_to_execute(self):
        """Test prepare_to_circuits_to_excute"""

        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.h(1)

        stateprep_circuit = QuantumCircuit(4)
        stateprep_circuit.compose(circuit, [0, 1], inplace=True)
        stateprep_circuit.compose(circuit, [2, 3], inplace=True)

        ansatz = EfficientSU2(2, entanglement="linear")

        initial_point = np.random.rand(len(ansatz.parameters))
        circuits_to_execute = prepare_circuits_to_execute(
            stateprep_circuit, ansatz, initial_point
        )

        for idx, circuit in enumerate(circuits_to_execute):
            if idx == 0:
                self.assertTrue("dip_test_circuit" in circuit.name)
            else:
                self.assertTrue("pdip_test_circuit" in circuit.name)
