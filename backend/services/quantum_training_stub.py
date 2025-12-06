# backend/services/quantum_training_stub.py
"""
Small prototype showing how to run a variational circuit optimization on quantum_features.npy.
This is for demo/prototyping only. For production training you'll want batched training,
proper loss, batching, and possibly a hybrid classical optimizer loop.
"""
import numpy as np, json, time
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def load_quantum_data():
    X = np.load(DATA_DIR / "quantum_features.npy")
    y = np.load(DATA_DIR / "quantum_labels.npy")
    return X, y

def simple_quantum_eval(X_sample, n_qubits=3):
    # VERY SMALL demo: map first n_qubits dims to rotation angles and compute simple score
    try:
        from qiskit import QuantumCircuit
        from qiskit.providers.aer import AerSimulator
    except Exception:
        # fallback: classical pseudo-eval
        s = float(np.abs(X_sample[:n_qubits]).sum())
        return max(0.0, min(1.0, 1.0 - s/ (n_qubits*2.0)))

    qc = QuantumCircuit(n_qubits)
    # encode angles
    for i in range(n_qubits):
        theta = float(X_sample[i] * np.pi)
        qc.ry(theta, i)
    for i in range(n_qubits-1):
        qc.cz(i, i+1)
    qc.measure_all()
    sim = AerSimulator()
    result = sim.run(qc, shots=256).result()
    counts = result.get_counts()
    # compute simple score from |0...0> frequency
    zero_state = "0"*n_qubits
    freq = counts.get(zero_state, 0) / 256.0
    return freq

def demo_run(n_samples=10):
    X, y = load_quantum_data()
    print("Loaded quantum data:", X.shape)
    out = []
    for i in range(min(n_samples, X.shape[0])):
        s = simple_quantum_eval(X[i])
        out.append({"idx": i, "label": int(y[i]), "score": float(s)})
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    demo_run()
