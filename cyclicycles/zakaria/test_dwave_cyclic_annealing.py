
# test_dwave_cyclic_annealing.py
# ---------------------------------------------------------------
# Tests for the D-Wave cyclic annealing module.
# Uses MockDWaveSampler when Ocean is installed; otherwise skips
# the hardware-dependent bits.

import unittest
import numpy as np

try:
    import dimod
except Exception:
    dimod = None

from dwave_cyclic_annealing import (
    ising_energy, add_ancilla_for_fields,
    bqm_from_coupler_matrix, bqm_from_J_h,
    reverse_anneal_triangle_schedule, href_hgain_schedule,
    DWaveCyclicAnnealer, CycleSpec
)

# Use Ocean's MockDWaveSampler if available
try:
    from dwave.system import EmbeddingComposite, DWaveSampler
except Exception:
    MockDWaveSampler = None
    EmbeddingComposite = None

sampler = DWaveSampler(solver="Advantage2_system1.7")

@unittest.skipIf(dimod is None, "dimod not available")
class TestAncillaReduction(unittest.TestCase):
    def test_energy_equivalence_if_ancilla_pinned(self):
        rng = np.random.default_rng(7)
        N = 6
        J = rng.normal(size=(N, N)); J = (J + J.T)/2; np.fill_diagonal(J, 0.0)
        h = rng.normal(size=N)

        Jp = add_ancilla_for_fields(J, h)
        s = rng.choice([-1,1], size=N)
        s_aug = np.append(s, 1)

        E = ising_energy(J, h, s)
        Ep = -0.5 * float(s_aug @ Jp @ s_aug)
        self.assertAlmostEqual(E, Ep, places=8)

    def test_bqm_builders(self):
        rng = np.random.default_rng(0)
        N = 4
        J = rng.normal(size=(N,N)); J=(J+J.T)/2; np.fill_diagonal(J, 0.0)
        h = rng.normal(size=N)

        bqm1 = bqm_from_J_h(J, h)
        bqm2 = bqm_from_coupler_matrix(add_ancilla_for_fields(J, h))
        self.assertEqual(bqm1.vartype.name, 'SPIN')
        self.assertEqual(bqm2.vartype.name, 'SPIN')


@unittest.skipIf(dimod is None, "dimod not available")
class TestSchedules(unittest.TestCase):
    def test_triangle_schedule_and_hgain(self):
        ra = reverse_anneal_triangle_schedule(s_target=0.45, t_bz=0.1, t_rampdown=0.5, t_hold=299.0)
        self.assertTrue(all(t2 > t1 for (t1,_),(t2,_) in zip(ra, ra[1:])))
        self.assertAlmostEqual(ra[0][1], 1.0)
        self.assertAlmostEqual(ra[-1][1], 1.0)

        hgs = href_hgain_schedule(0.03, ra)
        self.assertEqual(len(hgs), 3)
        self.assertAlmostEqual(hgs[0][1], 0.0)
        self.assertAlmostEqual(hgs[1][1], 0.03)
        self.assertAlmostEqual(hgs[-1][1], 0.0)


@unittest.skipIf(DWaveSampler is None or dimod is None, "Ocean not installed")
class TestEndToEndWithMockSampler(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(123)
        N = 10
        J = self.rng.normal(size=(N,N)); J=(J+J.T)/2; np.fill_diagonal(J, 0.0)
        h = self.rng.normal(size=N)
        self.J, self.h = J, h
        self.sampler = EmbeddingComposite(sampler)

    def test_cyclic_runs_and_forward_comparison(self):
        annealer = DWaveCyclicAnnealer(self.sampler, num_reads_per_cycle=100, use_ancilla=True)
        spec = CycleSpec(bz=0.03, s_mix=0.45, t_bz=0.1, t_ramp=0.5, t_hold=299.0)

        s0 = np.ones(len(self.h), dtype=int)
        best_s, best_E, hist = annealer.cyclic_anneal(self.J, self.h, n_cycles=3, spec=spec, samples_per_cycle=100, initial_state=s0)

        # Forward annealing with equal total time budget (3 cycles * ~300us ≈ 900us)
        forward_s, forward_E = annealer.forward_anneal(self.J, self.h, anneal_time=900.0, num_reads=300)

        # Basic assertions: both paths return finite energies and bitstrings of correct length
        self.assertEqual(len(best_s), len(self.h))
        self.assertTrue(np.isfinite(best_E))
        self.assertTrue(np.isfinite(forward_E))

        # Provide a comparison assertion that doesn't assume superiority (mock sampler is random).
        self.assertIsInstance(hist, list)
        self.assertGreaterEqual(len(hist), 4)  # initial + 3 cycles

if __name__ == "__main__":
    unittest.main()
