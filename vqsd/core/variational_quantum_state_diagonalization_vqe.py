"""Variation Quantum State Diagonilzation."""

from time import time
from typing import List, Callable, Union, Tuple

import numpy as np
from qiskit import transpile
from qiskit.algorithms import VQE
from qiskit.utils.mitigation import complete_meas_cal
from qiskit.quantum_info import Statevector
from qiskit.utils import QuantumInstance
from qiskit import QuantumCircuit
from qiskit.circuit import Instruction
from qiskit.result import Result
from qiskit.opflow import OperatorBase, StateFn


from vqsd.core.VariationalQuantumStateDiagonilzationConfig import (
    VariationalQuantumStateDiagonilzationConfig,
)
from vqsd.utils.subroutines import (
    prepare_circuits_to_execute,
    eval_tests_with_result,
    get_optimizer_instance,
)
from vqsd.utils.execution_subroutines import execute_with_retry

from vqsd.utils.logger import Logger as logger
from vqsd.utils.meas_mit_fitters_faster import CompleteMeasFitter


class VariationalQuantumStateDiagonilzation(VQE):
    # pylint: disable=too-many-instance-attributes
    """
    TODO
    """

    def __init__(
        self,
        ansatz: QuantumCircuit,
        initial_state: Union[
            str,
            dict,
            Result,
            list,
            np.ndarray,
            Statevector,
            QuantumCircuit,
            Instruction,
            OperatorBase,
        ],
        config: VariationalQuantumStateDiagonilzationConfig,
    ) -> None:

        super().__init__(
            ansatz=ansatz,
            optimizer=get_optimizer_instance(config),
            initial_point=config.initial_params,
            max_evals_grouped=config.max_evals_grouped,
            callback=None,
        )

        self._ansatz = ansatz
        self._config = config
        self._shots = config.shots
        self._weight = config.weight
        self._eps_max = config.eps_max
        self._initial_state = StateFn(initial_state)
        self._num_qubits = self._initial_state.num_qubits
        self._initial_state_circuit = self._initial_state.to_circuit_op().to_circuit()
        self._purity = 1.0

        self._energy_each_iteration_each_paramset: List[float] = []
        self._paramsets_each_iteration: List[np.ndarray] = []
        self._zero_noise_extrap = config.zero_noise_extrap

        self._iteration_start_time = np.nan

        self._backend = config.backend
        self.quantum_instance = QuantumInstance(backend=self._backend)
        self._initial_layout = config.qubit_layout
        self._shots = config.shots
        self._meas_error_mit = config.meas_error_mit
        self._meas_error_shots = config.meas_error_shots
        self._meas_error_refresh_period_minutes = (
            config.meas_error_refresh_period_minutes
        )
        self._meas_error_refresh_timestamp = None
        self._coupling_map = self._backend.configuration().coupling_map
        self._meas_fitter = None
        self._rep_delay = config.rep_delay

        # self.aux_results: List[Tuple[str, AuxiliaryResults]] = []

        self.parameter_sets: List[np.ndarray] = []
        self.energy_mean_each_parameter_set: List[float] = []
        self.energy_std_each_parameter_set: List[float] = []

        # Paramters for get_energy_evaluation. Moved them here to match parent function signature
        self._shots_multiplier = 1
        self._bootstrap_trials = 0

        statevector_sims = ["aer_simulator_statevector", "statevector_simulator"]
        if self._backend.name() in statevector_sims:
            self._is_sv_sim = True
        else:
            self._is_sv_sim = False

        if self._ansatz.num_qubits > self._num_qubits:
            raise ValueError(
                "The number of qubits in ansatz does "
                "not match the number of bits in initial_state."
            )

        if 0 <= self._weight <= 1.0:
            raise ValueError("The hyper parameter q must be in the range [0,1]")

        # pylint: disable=fixme
        # TODO: Validate whether there are measurements in state
        # if self._initial_state.is_measurement:
        #     raise ValueError("The initial state cannot measurements.")

        self._stateprep_circuit = QuantumCircuit(
            2 * self._num_qubits, name="stateprep_circuit"
        )

        self._stateprep_circuit.compose(
            self._initial_state_circuit, [0, self._num_qubits]
        )
        self._stateprep_circuit.compose(
            self._initial_state_circuit[self._num_qubits, 2 * self._num_qubits]
        )

    @property
    def ansatz(self):
        """
        TODO
        """
        return self._ansatz

    @ansatz.setter
    def ansatz(self, ansatz: QuantumCircuit):
        """
        TODO
        """
        self._ansatz = ansatz

    def get_energy_evaluation(
        self,
        operator: OperatorBase,
        return_expectation: bool = False,
    ) -> Union[
        Callable[[np.ndarray], Union[float, List[float]]],
        Tuple[Callable[[np.ndarray], Union[float, List[float]]], float],
    ]:
        """
        TODO
        """

        if self._is_sv_sim:
            self._shots_multiplier = 1

        ansatz_params = self.ansatz.parameters
        _, expectation = self.construct_expectation(
            ansatz_params, operator, return_expectation=True
        )

        def energy_evaluation(parameters):
            # pylint: disable=too-many-locals
            """
            TODO
            """
            logger.log("------ new iteration energy evaluation -----")
            parameter_sets = np.reshape(parameters, (-1, self._ansatz.num_parameters))
            self.parameter_sets.append(parameter_sets)

            new_iteration_start_time = time.time()

            logger.log(
                "duration of last iteration:",
                new_iteration_start_time - self._iteration_start_time,
            )

            self._iteration_start_time = new_iteration_start_time

            logger.log("Parameter sets:", parameter_sets)

            eval_result = self._cost_evaluation(parameter_sets)

            (
                local_obj_fun_eval_mean_each_parameter_set,
                local_obj_fun_eval_std_each_parameter_set,
                global_obj_fun_eval_mean_each_parameter_set,
                global_obj_fun_eval_std_each_parameter_set,
                weighted_obj_fun_eval_mean_each_parameter_set,
                weighted_obj_fun_eval_std_each_parameter_set,
            ) = eval_result

            self._energy_each_iteration_each_paramset.append(
                weighted_obj_fun_eval_mean_each_parameter_set
            )
            self._paramsets_each_iteration.append(parameter_sets)

            for (
                params,
                local_cost_mean,
                local_cost_std,
                global_cost_mean,
                global_cost_std,
                weighted_cost_mean,
                weighted_cost_std,
            ) in zip(
                parameter_sets,
                local_obj_fun_eval_mean_each_parameter_set,
                local_obj_fun_eval_std_each_parameter_set,
                global_obj_fun_eval_mean_each_parameter_set,
                global_obj_fun_eval_std_each_parameter_set,
                weighted_obj_fun_eval_mean_each_parameter_set,
                weighted_obj_fun_eval_mean_each_parameter_set,
            ):
                logger.log("local cost mean:", local_cost_mean)
                logger.log("local cost std:", local_cost_std)
                logger.log("global cost mean:", global_cost_mean)
                logger.log("global cost std:", global_cost_std)

                self._eval_count += 1
                if self._callback is not None:
                    self._callback(
                        self._eval_count, params, weighted_cost_mean, weighted_cost_std
                    )

            self.energy_mean_each_parameter_set.append(
                weighted_obj_fun_eval_mean_each_parameter_set
            )
            self.energy_std_each_parameter_set.append(
                weighted_obj_fun_eval_std_each_parameter_set
            )

            return (
                np.array(weighted_obj_fun_eval_mean_each_parameter_set)
                if len(weighted_obj_fun_eval_mean_each_parameter_set) > 1
                else weighted_obj_fun_eval_mean_each_parameter_set[0]
            )

        if return_expectation:
            return energy_evaluation, expectation

        return energy_evaluation

    def _cost_evaluation(self, parameter_sets):
        # pylint: disable=too-many-locals
        """
        TODO
        """
        circuits_to_execute = []
        for _, params in enumerate(parameter_sets):
            logger.log("Constructing the circuits for parameter set", params, "...")
            circuits_to_execute = prepare_circuits_to_execute(
                self._stateprep_circuit, self._ansatz, params, self._is_sv_sim
            )

        logger.log("Transpiling circuits...")
        logger.log(self._initial_layout)

        circuits_to_execute = transpile(
            circuits_to_execute,
            self._backend,
            initial_layout=self._initial_layout,
            coupling_map=self._coupling_map,
        )

        if self._meas_error_mit:
            if (not self._meas_fitter) or (
                (time.time() - self._meas_error_refresh_timestamp) / 60
                > self._meas_error_refresh_period_minutes
            ):
                logger.log("Generating measurement fitter...")
                physical_qubits = np.asarray(self._initial_layout).tolist()
                cal_circuits, state_labels = complete_meas_cal(
                    np.arange(len(physical_qubits)).tolist()
                )
                result = execute_with_retry(
                    cal_circuits, self._backend, self._meas_error_shots, self._rep_delay
                )
                self._meas_fitter = CompleteMeasFitter(result, state_labels)
                self._meas_error_refresh_timestamp = time.time()

        result = execute_with_retry(
            circuits_to_execute,
            self._backend,
            self._shots * self._shots_multiplier,
            self._rep_delay,
        )

        if self._meas_error_mit:
            logger.log("Applying meas fitter/filter...")
            result = self._meas_fitter.filter.apply(result)

        logger.log("Done executing. Analyzing results...")

        local_obj_fun_eval_mean_each_parameter_set = [0] * len(parameter_sets)
        local_obj_fun_eval_std_each_parameter_set = [0] * len(parameter_sets)

        global_obj_fun_eval_mean_each_parameter_set = [0] * len(parameter_sets)
        global_obj_fun_eval_std_each_parameter_set = [0] * len(parameter_sets)

        weighted_obj_fun_eval_mean_each_parameter_set = [0] * len(parameter_sets)
        weighted_obj_fun_eval_std_each_parameter_set = [0] * len(parameter_sets)

        for idx, params in enumerate(parameter_sets):
            for _, res in enumerate(result):
                pdip_eval, dip_eval = eval_tests_with_result(res, self._is_sv_sim)
                local_obj_fun_eval = self._purity - pdip_eval
                global_obj_fun_eval = self._purity - dip_eval
                weighted_obj_func_eval = (
                    self._weight * global_obj_fun_eval + (1 - self._weight) * local_obj_fun_eval
                )

                local_obj_fun_eval_mean_each_parameter_set[idx] = local_obj_fun_eval
                global_obj_fun_eval_mean_each_parameter_set[idx] = global_obj_fun_eval
                weighted_obj_fun_eval_mean_each_parameter_set[
                    idx
                ] = weighted_obj_func_eval

        return (
            local_obj_fun_eval_mean_each_parameter_set,
            local_obj_fun_eval_std_each_parameter_set,
            global_obj_fun_eval_mean_each_parameter_set,
            global_obj_fun_eval_std_each_parameter_set,
            weighted_obj_fun_eval_mean_each_parameter_set,
            weighted_obj_fun_eval_std_each_parameter_set,
        )
