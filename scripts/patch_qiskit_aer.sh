#!/usr/bin/env bash
set -euo pipefail
ROOT="$(pwd)"

files=(
  "backend/app/services/quantum_service.py"
  "backend/ml/train_quantum_vqc.py"
)

echo "Running patch from: $ROOT"
for f in "${files[@]}"; do
  if [ ! -f "$f" ]; then
    echo "Skipping (not found): $f"
    continue
  fi

  bak="${f}.bak.$(date +%s)"
  echo "Backing up $f -> $bak"
  cp "$f" "$bak"

  python - <<'PY'
import io,sys,re,os
path = sys.argv[1]
s = open(path, "r", encoding="utf-8").read()

orig = s

# 1) Replace single-line qiskit import that includes Aer:
# from qiskit import QuantumCircuit, Aer, transpile, execute
s = re.sub(
    r"from\s+qiskit\s+import\s+([^\\n]*\bAer\b[^\\n]*)",
    lambda m: re.sub(r"\bAer\b\s*,?", "", m.group(0)).replace("from qiskit import", "from qiskit import"),
    s
)

# If the file had "from qiskit import QuantumCircuit, Aer, transpile, execute" -> remove Aer only
s = re.sub(r"\b,\s*Aer\b", "", s)
s = re.sub(r"\bAer\s*,\s*", "", s)

# 2) Ensure we import qiskit basics if missing (QuantumCircuit/transpile/execute/Parameter/COBYLA)
# We'll add the robust Aer import block near the top after the qiskit imports or at top of file.
add_block = '''
# ---- patched for qiskit_aer compatibility ----
# Aer may be installed as a separate package (qiskit_aer). Try to import it defensively.
try:
    # new Aer provider package (qiskit-aer)
    from qiskit_aer import AerSimulator
    AER_AVAILABLE = True
except Exception:
    AerSimulator = None
    AER_AVAILABLE = False
# ---------------------------------------------
'''

# Insert add_block after the first occurrence of "from qiskit import" or at top if not found
if "from qiskit import" in s:
    idx = s.find("from qiskit import")
    # find end of that import line
    endline = s.find("\n", idx)
    if endline == -1:
        endline = idx
    insert_pos = endline + 1
    # but avoid inserting twice if block already present
    if "patched for qiskit_aer compatibility" not in s:
        s = s[:insert_pos] + add_block + s[insert_pos:]
else:
    if "patched for qiskit_aer compatibility" not in s:
        s = add_block + s

# 3) Replace Aer.get_backend("...") calls with AerSimulator()
s = re.sub(r"\bAer\.get_backend\s*\([^)]*\)", "AerSimulator()", s)
s = re.sub(r"\bAer\.get_backend\s*\(\s*['\"][^'\"]+['\"]\s*\)", "AerSimulator()", s)

# 4) Also replace 'from qiskit.providers.aer import Aer' or similar if present
s = re.sub(r"from\s+qiskit\.providers\.aer\s+import\s+[^\n]+\n", "", s)

# 5) If file uses 'Aer' variable elsewhere (e.g., Aer.run), try to replace "Aer." occurrences for backend related calls:
# Only transform common patterns used in this repo; avoid global blind replacement.
s = re.sub(r"\bAer\.run\(", "AerSimulator().run(", s)
s = re.sub(r"\bAer\.simulate\(", "AerSimulator().simulate(", s)

# Write only if changed
if s != orig:
    open(path, "w", encoding="utf-8").write(s)
    print("Patched:", path)
else:
    print("No changes needed:", path)
PY "$f"

done

echo "Patch complete. You should:"
echo "  1) activate venv:  source backend/.venv/bin/activate"
echo "  2) restart server or run training: python backend/ml/train_quantum_vqc.py"
echo "If something looks wrong, restore from the backup files (*.bak.*)."
