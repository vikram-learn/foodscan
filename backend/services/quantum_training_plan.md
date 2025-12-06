# Quantum training & dataset plan (FoodScan-X)

## Goals
- Train a hybrid quantum/classical model that helps recommend portion safety/risk scoring using:
  - Food-101 images (visual),
  - Nutrition information (USDA) if available,
  - Ayurveda-derived features.

## Data sources
- Primary: Food-101 (images + classes).
- Secondary (optional): USDA FoodData Central for per-food nutrition lookup.
- Labels: map Food-101 classes to nutrition defaults (approx) or use USDA to enrich.

## Preprocessing
1. Use `preprocess.py` to create color-histogram features (fast).
2. Use `train_classical.py` to get baseline results.
3. Use `prepare_quantum_data.py` to reduce feature dimension (PCA -> D ≤ 8).

## Quantum approach
- Start with simulation-based experiments:
  - Variational Quantum Classifier (VQC) with parameterized rotations and small entangling layers (2–4 qubits).
  - Optimize with COBYLA / SPSA or classical optimizers (scipy).
- Hybrid strategy:
  - Train classical feature extractor (pretrained CNN) to produce low-dim embeddings.
  - Feed embeddings into quantum circuit (quantum layer) for final decision.

## Training pipeline (recommended)
1. Precompute features for whole dataset (fast, CPU).
2. Prepare quantum features (PCA -> D).
3. Train quantum model on small subset (N ≲ 1000) on AerSimulator.
4. Compare to classical baseline (RandomForest).
5. If promising, iterate: increase samples, augment embeddings (use CNN), try hybrid nets.

## Compute & tooling
- Local dev: Qiskit Aer (simulator) — small circuits only.
- For real quantum hardware: IBMQ account + noise-aware transpilation; batch jobs queued.
- Storage: keep features `.npy` under `backend/data/` for fast iteration.

## Evaluation
- Classification accuracy (if classes) or regression / risk score MSE.
- Robustness under noise (simulate noisy Aer) before hardware runs.
- Human-readable explanations (dosha effects, nutrition).

## Next actions (short)
- Finish dataset download.
- Run `preprocess.py`, `train_classical.py`, `prepare_quantum_data.py`.
- Run the quantum stub to verify Aer runs locally.
- If Aer errors appear, I can help patch imports / provider configuration.

