# Grover's Algorithm – Quantum Search Simulation

An implementation and simulation of **Grover's Quantum Search Algorithm** using Python and Qiskit. This project demonstrates how quantum computing can accelerate unstructured search by finding a marked state in an unsorted search space.

## Overview

Grover's Algorithm is a quantum search algorithm that provides a quadratic speedup over classical unstructured search.

While a classical search requires approximately \(O(N)\) queries in the worst case, Grover's Algorithm requires approximately \(O(\sqrt{N})\) oracle queries.

This project simulates a search across 8 possible states using 3 qubits, with the target state set to `|101⟩`.

## Key Features

- Implementation of Grover's Quantum Search Algorithm.
- Quantum circuit construction using Qiskit.
- Oracle implementation to mark the target state.
- Diffusion operator for amplitude amplification.
- Configurable number of measurement shots.
- Circuit visualization and simulation.
- Graphical output and result analysis using Matplotlib.

## Algorithm Configuration

| Parameter | Value |
|---|---|
| Number of Qubits | 3 |
| Search Space | 8 states |
| Target State | `101` |
| Optimal Grover Iterations | 2 |
| Default Measurement Shots | 4096 |

## How It Works

The algorithm follows these steps:

1. **Initialization:** Three qubits are initialized in the state `|000⟩`.
2. **Superposition:** Hadamard gates create an equal superposition of all 8 possible states.
3. **Oracle:** The oracle identifies and marks the target state `|101⟩` by changing its phase.
4. **Diffusion Operator:** Amplitude amplification increases the probability of measuring the marked state.
5. **Measurement:** The qubits are measured to obtain the final probability distribution.

## Technologies Used

- Python
- Qiskit
- Matplotlib
- NumPy

## Installation and Execution

### 1. Clone the Repository

```bash
git clone https://github.com/annie-bliss/Grover-s-Algorithm.git
```

### 2. Navigate to the Project Directory

```bash
cd Grover-s-Algorithm
```

### 3. Create a Virtual Environment (Optional)

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install qiskit matplotlib numpy
```

### 5. Run the Simulation

```bash
python grover.py --shots 4096
```

You can modify the number of measurement shots:

```bash
python grover.py --shots 8192
```

## Project Structure

```text
Grover-s-Algorithm/
│
├── grover.py          # Main implementation
├── output.txt         # Saved terminal output
├── grover_output/     # Generated figures and visualizations
└── README.md          # Project documentation
```

## Expected Outcome

The simulation aims to amplify the probability of measuring the marked state `|101⟩`.

With the optimal number of Grover iterations, the target state should have a significantly higher measurement probability than the other states.

The project also generates visualizations to help understand the behavior of the quantum search algorithm.

## Learning Outcomes

- Understanding quantum superposition and interference.
- Exploring quantum oracles and phase inversion.
- Understanding amplitude amplification.
- Constructing and simulating quantum circuits.
- Analyzing quantum measurement probabilities.

## Author

**Ankita Kumari**

GitHub: [@annie-bliss](https://github.com/annie-bliss)

---

*Exploring the potential of quantum computing through practical implementation and simulation.*
