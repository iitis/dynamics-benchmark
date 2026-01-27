import numpy as np
from dwave.system import DWaveSampler, EmbeddingComposite
from dwave_cyclic_annealing import DWaveCyclicAnnealer, CycleSpec, ising_energy

N=12

def build_ferromagnetic_chain(N, Jval=1.0):
    # J is NxN, symmetric, nearest-neighbor couplings only
    J = np.zeros((N, N), dtype=float)
    for i in range(N-1):
        J[i, i+1] = J[i+1, i] = Jval
    h = np.zeros(N, dtype=float)
    return J, h

J, h = build_ferromagnetic_chain(N)
def exact_ground_states_for_chain(N):
    # For J ferromagnetic chain and h=0, two ground states: all +1 and all -1
    s_plus = np.ones(N, dtype=int)
    s_minus = -np.ones(N, dtype=int)
    return [s_plus, s_minus]

 # Known ground states and energy
gs_states = exact_ground_states_for_chain(N)
E_star = ising_energy(J, h, gs_states[0])  # both have same energy
print(f"Known ground-state energy E* = {E_star:.3f} for N={N} (degenerate, ±all-aligned)")

qpu = DWaveSampler(solver="Advantage_system6.4")

sampler = EmbeddingComposite(qpu)  # requires Leap creds
annealer = DWaveCyclicAnnealer(sampler, use_ancilla=True)

spec = CycleSpec(bz=0.03, s_mix=0.45, t_bz=0.1, t_ramp=0.5, t_hold=299.0)
best_s, best_E, hist = annealer.cyclic_anneal(J, h, n_cycles=10, spec=spec, samples_per_cycle=10)

# Equal-time forward baseline (~10 * 300 µs)
f_s, f_E = annealer.forward_anneal(J, h, anneal_time=100.0, num_reads=10)
print("cyclic:", best_E, "forward:", f_E)
