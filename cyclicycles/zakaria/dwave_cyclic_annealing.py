
# dwave_cyclic_annealing.py
# ---------------------------------------------------------------
# Cyclic annealing on D-Wave hardware with optional ancilla reduction.
# This file avoids triple-quoted strings per user request; comments use '#'.

from dataclasses import dataclass
from typing import List, Sequence, Tuple, Optional
import dwave

import numpy as np

# Dimod is required for BQM construction. Ocean (dwave-system) is optional
# for running on a QPU; tests can use MockDWaveSampler.
try:
    import dimod
except Exception as e:
    raise ImportError("This module requires 'dimod'. Install via 'pip install dwave-ocean-sdk'.") from e


# ---------------------------- helpers ----------------------------

def ising_energy(J: np.ndarray, h: np.ndarray, s: np.ndarray) -> float:
    # Physics convention: E = -1/2 s^T J s - h^T s  for s in {-1,+1}
    s = s.astype(float)
    return -0.5 * float(s @ J @ s) - float(h @ s)


def add_ancilla_for_fields(J: np.ndarray, h: np.ndarray) -> np.ndarray:
    # Append ancilla spin so Hp has no one-body fields.
    # J'[:N,:N]=J, J'[i,N]=J'[N,i]=h_i, J'[N,N]=0
    N = len(h)
    Jp = np.zeros((N + 1, N + 1), dtype=float)
    Jp[:N, :N] = J
    Jp[:N, N] = h
    Jp[N, :N] = h
    return Jp


def bqm_from_coupler_matrix(J: np.ndarray) -> dimod.BinaryQuadraticModel:
    # Convert dense Ising coupler matrix (pairwise only) to a SPIN BQM with zero linear biases.
    N = J.shape[0]
    linear = {i: 0.0 for i in range(N)}
    quadratic = {}
    for i in range(N):
        for j in range(i + 1, N):
            val = float(J[i, j])
            if val != 0.0:
                quadratic[(i, j)] = val
    return dimod.BinaryQuadraticModel(linear, quadratic, 0.0, vartype=dimod.SPIN)


def bqm_from_J_h(J: np.ndarray, h: np.ndarray) -> dimod.BinaryQuadraticModel:
    # Build a SPIN BQM from physics-convention J,h where
    # E = -sum_{i<j} J_ij s_i s_j - sum_i h_i s_i.
    # dimod uses E = sum_i b_i s_i + sum_{i<j} J_ij s_i s_j, so we negate.
    N = len(h)
    linear = {i: -float(h[i]) for i in range(N)}
    quadratic = {}
    for i in range(N):
        for j in range(i + 1, N):
            val = -float(J[i, j])
            if val != 0.0:
                quadratic[(i, j)] = val
    return dimod.BinaryQuadraticModel(linear, quadratic, 0.0, vartype=dimod.SPIN)


def add_reference_field(bqm: dimod.BinaryQuadraticModel,
                        s_ref: np.ndarray,
                        include_ancilla: bool = False) -> None:
    # Encode H_ref = -sum_i s_i^r sigma_i^z as linear biases in-place.
    # Ancilla gets no reference bias unless include_ancilla=True.
    N = len(s_ref)
    last = N if include_ancilla else N - 1
    for i in range(last):
        bqm.set_linear(i, bqm.get_linear(i) - int(s_ref[i]))


def reverse_anneal_triangle_schedule(
    s_target: float,
    t_bz: float = 0.1,
    t_rampdown: float = 0.5,
    t_hold: float = 299.0
) -> List[Tuple[float, float]]:
    # Build (t, s) schedule for reverse annealing:
    # start at s=1, stay until t_bz (while Bz ramps via h_gain),
    # ramp to s_target in t_rampdown, hold for t_hold, return to s=1 at the end.
    if not (0.0 < s_target < 1.0):
        raise ValueError("s_target must be in (0,1)")
    t0 = 0.0
    t1 = t_bz
    t2 = t_bz + t_rampdown
    t3 = t_bz + t_rampdown + t_hold
    return [(t0, 1.0), (t1, 1.0), (t2, s_target), (t3, s_target), (t3 + 0.4, 1.0)]  # ~300 µs total


def href_hgain_schedule(bz: float, anneal_sched: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    # (t, g) schedule for h_gain_schedule aligned with reverse_anneal schedule.
    if bz < 0:
        raise ValueError("bz must be non-negative")
    if len(anneal_sched) < 3:
        raise ValueError("anneal schedule too short")
    t0 = anneal_sched[0][0]
    t1 = anneal_sched[1][0]
    t_end = anneal_sched[-1][0]
    return [(t0, 0.0), (t1, float(bz)), (t_end, 0.0)]


@dataclass
class CycleSpec:
    # Parameters for one cyclic anneal on hardware
    bz: float
    s_mix: float
    t_bz: float = 0.1
    t_ramp: float = 0.5
    t_hold: float = 299.0

    def build_schedules(self) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        # Return (reverse_anneal_schedule, h_gain_schedule)
        ra = reverse_anneal_triangle_schedule(self.s_mix, self.t_bz, self.t_ramp, self.t_hold)
        hg = href_hgain_schedule(self.bz, ra)
        return ra, hg


class DWaveCyclicAnnealer:
    # Controller for cyclic annealing on a D-Wave sampler (QPU or mock/local).

    def __init__(
        self,
        sampler,
        chain_strength: Optional[float] = None,
        num_reads_per_cycle: int = 100,
        reinitialize_state: bool = True,
        random_seed: Optional[int] = None,
        use_ancilla: bool = True
    ) -> None:
        self.sampler = sampler
        self.chain_strength = chain_strength
        self.num_reads_per_cycle = int(num_reads_per_cycle)
        self.reinitialize_state = bool(reinitialize_state)
        self.rng = np.random.default_rng(random_seed)
        self.use_ancilla = bool(use_ancilla)

    # ----------------- problem compilation -----------------

    def _problem_bqm(self, J: np.ndarray, h: np.ndarray) -> Tuple[dimod.BinaryQuadraticModel, Optional[int]]:
        # Compile the base problem BQM used in the cycle.
        # If use_ancilla=True: build coupler-only Hp with ancilla; else: use original (J,h) directly.
        if self.use_ancilla:
            Jp = add_ancilla_for_fields(J, h)    # pairwise-only
            bqm = bqm_from_coupler_matrix(Jp)    # zero linear terms
            anc = Jp.shape[0] - 1
            return bqm, anc
        else:
            bqm = bqm_from_J_h(J, h)             # includes local fields
            return bqm, None

    # ----------------- cycle driver -----------------

    def cycle_once(
        self,
        J: np.ndarray,
        h: np.ndarray,
        s_ref: np.ndarray,
        spec: CycleSpec
    ) -> Tuple[np.ndarray, float]:
        # Execute a single cyclic annealing run and return (best_state, best_energy) in original energy.
        N = len(h)
        bqm, anc = self._problem_bqm(J, h)

        # Add reference field only on original spins (ancilla excluded).
        s_ref_aug = np.append(s_ref, 1) if anc is not None else s_ref.copy()
        add_reference_field(bqm, s_ref_aug, include_ancilla=False)

        # Build schedules
        reverse_sched, h_gain = spec.build_schedules()

        # Initial state dictionary
        init = {i: int(s_ref[i]) for i in range(N)}
        if anc is not None:
            init[anc] = 1

        kwargs = dict(
            num_reads=self.num_reads_per_cycle,
            initial_state=init,
            reinitialize_state=self.reinitialize_state,
            anneal_schedule=reverse_sched,
            h_gain_schedule=h_gain,
        )
        if self.chain_strength is not None:
            kwargs['chain_strength'] = self.chain_strength

        sampleset = self.sampler.sample(bqm, **kwargs)
        #dwave.inspector.show(sampleset)
        # Score by the original (J,h) energy ignoring ancilla
        bqm_original = bqm_from_J_h(J, h)
        best_s = None
        best_E = float('inf')
        for d in sampleset.data(['sample']):
            sample = d[0]
            sub = {i: sample[i] for i in range(N)}
            E = bqm_original.energy(sub)
            if E < best_E:
                best_E = E
                best_s = np.array([sub[i] for i in range(N)], dtype=int)
        return best_s, best_E

    def cyclic_anneal(
        self,
        J: np.ndarray,
        h: np.ndarray,
        n_cycles: int,
        spec: CycleSpec,
        samples_per_cycle: Optional[int] = None,
        initial_state: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float, List[float]]:
        # Multiple cycles; after each cycle, update reference to the best-of-cycle state.
        if samples_per_cycle is not None:
            self.num_reads_per_cycle = int(samples_per_cycle)

        N = len(h)
        s_ref = np.ones(N, dtype=int) if initial_state is None else np.asarray(initial_state, dtype=int)

        best_s = s_ref.copy()
        best_E = ising_energy(J, h, best_s)
        history = [best_E]

        for _ in range(n_cycles):
            cand_s, cand_E = self.cycle_once(J, h, best_s, spec)
            if cand_E < best_E:
                best_E = cand_E
                best_s = cand_s
            history.append(best_E)
        return best_s, best_E, history

    # ----------------- forward annealing (baseline) -----------------

    def forward_anneal(
        self,
        J: np.ndarray,
        h: np.ndarray,
        anneal_time: float = 300.0,
        num_reads: int = 1000
    ) -> Tuple[np.ndarray, float]:
        # Run a single forward anneal on the original (J,h).
        bqm = bqm_from_J_h(J, h)
        ss = self.sampler.sample(bqm, num_reads=num_reads, annealing_time=anneal_time)
        # Best by original energy
        best_E = float('inf')
        best = None
        for d in ss.data(['sample']):
            s = {i: d[0][i] for i in bqm.variables}
            E = bqm.energy(s)
            if E < best_E:
                best_E = E
                best = np.array([s[i] for i in sorted(bqm.variables)], dtype=int)
        return best, best_E
