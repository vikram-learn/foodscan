cat > README.dev-snippet.md <<'MD'
# Quick dev commands (FoodScan-X backend)

Assume repo root, venv at `backend/.venv`.

## Start server (dev)
./scripts/dev_start.sh

## Diagnostics - check qiskit & run sync optimize
./backend/.venv/bin/python - <<'PY'
from app.services import quantum_service
print("module flags:", quantum_service._module_flags())
print("sync:", quantum_service.run_quantum_optimize_sync({
  "nutrition_estimate":{"calories":420,"carbs":60,"protein":8,"fat":10},
  "ayurveda":{"dosha_effect":{"kapha":"increase"}}
}))
PY

## API smoke tests
# sync optimize
curl -s -X POST "http://127.0.0.1:8000/api/v1/quantum/optimize" \
  -H "Content-Type: application/json" \
  -d '{"scan_result":{"nutrition_estimate":{"calories":420,"carbs":60,"protein":8,"fat":10},"ayurveda":{"dosha_effect":{"kapha":"increase"}}}}' | jq

# image upload
curl -s -X POST "http://127.0.0.1:8000/api/v1/image" \
  -F 'file=@/tmp/test_food.jpg' \
  -F 'user_email=vikram@example.com' | jq

## ML tasks
# train classical
./backend/.venv/bin/python backend/ml/train_classical.py

# prepare quantum data
./backend/.venv/bin/python backend/ml/prepare_quantum_data.py

# train VQC (uses Aer if present)
./backend/.venv/bin/python backend/ml/train_quantum_vqc.py

## Tests
# run upload test
( cd backend && ./.venv/bin/python tests/test_upload_scan.py )
MD
