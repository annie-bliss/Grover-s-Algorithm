"""
Grover's Search Algorithm with Qiskit + analysis of quantum parallelism via superposition.
Done by: Ankita Kumari, Dhanya J Nadig, Keerthana Dinesh and Zoya Fathima

Usage:
    python grover.py                 # n=3 qubits, marked state '101'
    python grover.py --n 4 --marked 1011

Terminal Commands: 
    python grover.py
    python grover.py --n 4 --marked 1011
    python grover.py --shots 4096
"""
import argparse
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grover_output")
os.makedirs(OUT, exist_ok=True)


# ----------------------------------------------------------------------------
# 1. Building blocks
# ----------------------------------------------------------------------------
def multi_controlled_z(qc: QuantumCircuit, n: int) -> None:
    """Phase-flip |11...1>  (H - MCX - H on the last qubit)."""
    if n == 1:
        qc.z(0)
        return
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)


def oracle(n: int, marked: str) -> QuantumCircuit:
    """Phase oracle: |x> -> -|x> if x == marked, else |x>."""
    qc = QuantumCircuit(n, name="Oracle")
    # Qiskit is little-endian: rightmost char of the bitstring is qubit 0
    zeros = [i for i, b in enumerate(reversed(marked)) if b == "0"]
    for q in zeros:
        qc.x(q)
    multi_controlled_z(qc, n)
    for q in zeros:
        qc.x(q)
    return qc


def diffuser(n: int) -> QuantumCircuit:
    """Grover diffusion operator  2|s><s| - I  (inversion about the mean)."""
    qc = QuantumCircuit(n, name="Diffuser")
    qc.h(range(n))
    qc.x(range(n))
    multi_controlled_z(qc, n)
    qc.x(range(n))
    qc.h(range(n))
    return qc


def optimal_iterations(n: int) -> int:
    return max(1, int(math.floor(math.pi / 4 * math.sqrt(2 ** n))))


def grover_circuit(n: int, marked: str, iters: int, measure: bool = True) -> QuantumCircuit:
    qc = QuantumCircuit(n, n if measure else 0, name="Grover")
    qc.h(range(n))                       # uniform superposition
    qc.barrier()
    orc, dif = oracle(n, marked), diffuser(n)
    for _ in range(iters):
        qc.compose(orc, inplace=True)
        qc.barrier()
        qc.compose(dif, inplace=True)
        qc.barrier()
    if measure:
        qc.measure(range(n), range(n))
    return qc


# ----------------------------------------------------------------------------
# 2. Helpers
# ----------------------------------------------------------------------------
def labels(n):
    return [format(i, f"0{n}b") for i in range(2 ** n)]


def amplitudes(qc: QuantumCircuit) -> np.ndarray:
    return Statevector(qc).data.real  # all amplitudes stay real in Grover


def bar_amp(ax, amps, n, marked, title, show_mean=True):
    labs = labels(n)
    colors = ["tab:red" if l == marked else "tab:blue" for l in labs]
    ax.bar(labs, amps, color=colors)
    ax.axhline(0, color="k", lw=0.8)
    if show_mean:
        m = amps.mean()
        ax.axhline(m, color="green", ls="--", lw=1.2, label=f"mean = {m:.3f}")
        ax.legend(loc="upper left", fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("amplitude")
    ax.tick_params(axis="x", rotation=60, labelsize=8)
    ax.set_ylim(-1.05, 1.05)


# ----------------------------------------------------------------------------
# 3. Figures
# ----------------------------------------------------------------------------
def draw_circuits(n, marked, K):
    full = grover_circuit(n, marked, K, measure=True)
    full.draw("mpl", fold=-1, filename=os.path.join(OUT, "circuit_full.png"))
    oracle(n, marked).draw("mpl", filename=os.path.join(OUT, "circuit_oracle.png"))
    diffuser(n).draw("mpl", filename=os.path.join(OUT, "circuit_diffuser.png"))
    plt.close("all")
    # text diagram for the terminal
    print("\n=== Grover circuit (text diagram) ===")
    print(full.draw("text", fold=120))


def superposition_analysis(n, marked):
    N = 2 ** n
    qc = QuantumCircuit(n)
    qc.h(range(n))
    sv = Statevector(qc)
    amps = sv.data.real
    print("\n=== Quantum parallelism via superposition ===")
    print(f"After H^(x{n}) on |0...0>:  |s> = (1/sqrt({N})) * sum_x |x>")
    print(f"  every basis state has amplitude 1/sqrt(N) = {1/math.sqrt(N):.4f}")
    print(f"  every basis state has probability 1/N     = {1/N:.4f}")
    print(f"  => the oracle is applied to all N = {N} inputs in ONE call (phase kickback)")
    print("  but a measurement returns only one x; Grover's interference is what")
    print("  amplifies the marked state so that it becomes the likely outcome.")

    fig, ax = plt.subplots(figsize=(7, 3.6))
    bar_amp(ax, amps, n, marked, f"Uniform superposition after Hadamard layer (n={n}, N={N})")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "superposition.png"), dpi=200)
    plt.close(fig)


def iteration_steps(n, marked):
    """Show H -> oracle -> diffuser for the first iteration."""
    qc = QuantumCircuit(n)
    qc.h(range(n))
    a0 = amplitudes(qc)
    qc.compose(oracle(n, marked), inplace=True)
    a1 = amplitudes(qc)
    qc.compose(diffuser(n), inplace=True)
    a2 = amplitudes(qc)

    fig, axs = plt.subplots(1, 3, figsize=(15, 3.8))
    bar_amp(axs[0], a0, n, marked, "1) After H: uniform superposition")
    bar_amp(axs[1], a1, n, marked, "2) After oracle: marked amplitude sign-flipped")
    bar_amp(axs[2], a2, n, marked, "3) After diffuser: inversion about the mean")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "iteration_steps.png"), dpi=200)
    plt.close(fig)

    idx = int(marked, 2)
    print("\n=== First Grover iteration: amplitude of the marked state ===")
    print(f"  after H       : {a0[idx]: .4f}  (P = {a0[idx]**2:.4f})")
    print(f"  after oracle  : {a1[idx]: .4f}  (P = {a1[idx]**2:.4f})  <- sign flip, P unchanged")
    print(f"  after diffuser: {a2[idx]: .4f}  (P = {a2[idx]**2:.4f})  <- amplified")


def evolution(n, marked, K):
    idx = int(marked, 2)
    ks = list(range(K + 1))
    ncols = min(len(ks), 4)
    nrows = math.ceil(len(ks) / ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), squeeze=False)
    print("\n=== Probability of the marked state per iteration ===")
    for k in ks:
        sv = Statevector(grover_circuit(n, marked, k, measure=False))
        probs = np.abs(sv.data) ** 2
        ax = axs[k // ncols][k % ncols]
        ax.bar(labels(n), probs, color=["tab:red" if l == marked else "tab:blue" for l in labels(n)])
        ax.set_ylim(0, 1.05)
        ax.set_title(f"k = {k}   P(marked) = {probs[idx]:.3f}", fontsize=10)
        ax.tick_params(axis="x", rotation=60, labelsize=7)
        print(f"  k = {k}:  P({marked}) = {probs[idx]:.4f}")
    for j in range(len(ks), nrows * ncols):
        axs[j // ncols][j % ncols].axis("off")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "amplitude_evolution.png"), dpi=200)
    plt.close(fig)


def run_simulation(n, marked, K, shots=1024):
    qc = grover_circuit(n, marked, K, measure=True)
    sim = AerSimulator()
    tqc = transpile(qc, sim)
    counts = sim.run(tqc, shots=shots).result().get_counts()
    print(f"\n=== AerSimulator results ({shots} shots, k = {K}) ===")
    for state, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        tag = "  <-- marked" if state == marked else ""
        print(f"  {state}: {c:5d}  ({c / shots:6.2%}){tag}")
    fig = plot_histogram(counts, title=f"Grover n={n}, marked={marked}, k={K}, shots={shots}",
                         figsize=(8, 4))
    fig.savefig(os.path.join(OUT, "histogram.png"), dpi=200, bbox_inches="tight")
    plt.close("all")
    top = max(counts, key=counts.get)
    assert top == marked, "Grover failed to amplify the marked state!"
    print(f"  Most frequent outcome = {top}  ✔ matches marked state")
    return counts


def success_curve(n, marked, K):
    N = 2 ** n
    theta = math.asin(1 / math.sqrt(N))
    ks = np.arange(0, 2 * K + 3)
    theory = np.sin((2 * ks + 1) * theta) ** 2
    idx = int(marked, 2)
    sim = [abs(Statevector(grover_circuit(n, marked, int(k), measure=False)).data[idx]) ** 2
           for k in ks]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(ks, theory, "-", label=r"theory  $\sin^2((2k+1)\theta)$")
    ax.plot(ks, sim, "o", mfc="none", label="Qiskit statevector")
    ax.axvline(K, color="r", ls="--", label=f"optimal k = {K}")
    ax.set_xlabel("Grover iterations k")
    ax.set_ylabel("P(marked state)")
    ax.set_title("Success probability vs iterations (over-rotation past optimum)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "success_vs_iterations.png"), dpi=200)
    plt.close(fig)
    err = np.max(np.abs(theory - np.array(sim)))
    print(f"\nMax |theory - simulation| over k = {err:.2e}")


def query_scaling():
    ns = np.arange(2, 21)
    N = 2.0 ** ns
    classical = N / 2                       # expected queries, unstructured search
    grover = np.floor(np.pi / 4 * np.sqrt(N))
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.semilogy(ns, classical, "o-", label="Classical  ~ N/2")
    ax.semilogy(ns, grover, "s-", label=r"Grover  ~ $\frac{\pi}{4}\sqrt{N}$")
    ax.set_xlabel("number of qubits n   (N = 2^n)")
    ax.set_ylabel("oracle queries")
    ax.set_title("Query complexity: quadratic speed-up")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "query_scaling.png"), dpi=200)
    plt.close(fig)

    print("\n=== Query complexity ===")
    print(f"{'n':>3} {'N':>8} {'classical (N/2)':>16} {'Grover k':>9}")
    for n_ in (2, 3, 4, 6, 8, 10, 16, 20):
        Nn = 2 ** n_
        print(f"{n_:>3} {Nn:>8} {Nn // 2:>16} {optimal_iterations(n_):>9}")


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3, help="number of search qubits")
    ap.add_argument("--marked", type=str, default=None, help="marked bitstring, e.g. 101")
    ap.add_argument("--shots", type=int, default=1024)
    args = ap.parse_args()

    n = args.n
    marked = args.marked or ("101" if n == 3 else "1" * (n // 2) + "0" * (n - n // 2))
    assert len(marked) == n and set(marked) <= {"0", "1"}, "marked must be an n-bit string"
    K = optimal_iterations(n)
    print(f"Grover search: n = {n} qubits, N = {2**n} states, marked = |{marked}>, "
          f"optimal iterations k = {K}")

    draw_circuits(n, marked, K)
    superposition_analysis(n, marked)
    iteration_steps(n, marked)
    evolution(n, marked, K)
    run_simulation(n, marked, K, args.shots)
    success_curve(n, marked, K)
    query_scaling()
    print(f"\nAll figures saved in: {OUT}")


if __name__ == "__main__":
    main()
