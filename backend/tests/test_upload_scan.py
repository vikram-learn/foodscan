# backend/tests/test_upload_scan.py
# tests/test_upload_scan.py (add at top)
import sys, os
# add repository/backend to sys.path so "from app..." works anywhere
here = os.path.abspath(os.path.dirname(__file__))    # backend/tests
backend_root = os.path.abspath(os.path.join(here, ".."))
if backend_root not in sys.path:
    sys.path.insert(0, backend_root)

from fastapi.testclient import TestClient
from app.main import app
import io, json, sys

client = TestClient(app)

def run_test():
    # small binary image (you already have /tmp/test_food.jpg; if not, adjust)
    local_file = "/tmp/test_food.jpg"

    try:
        with open(local_file, "rb") as f:
            files = {"file": ("test_food.jpg", f, "image/jpeg")}
            data = {"user_email": "vikram@example.com"}
            resp = client.post("/api/v1/image", files=files, data=data)
    except FileNotFoundError:
        # fallback: upload a tiny in-memory jpeg-like blob
        fake_jpeg = io.BytesIO(b"\xff\xd8\xff\xe0" + b"\x00" * 200 + b"\xff\xd9")
        files = {"file": ("test_food.jpg", fake_jpeg, "image/jpeg")}
        data = {"user_email": "vikram@example.com"}
        resp = client.post("/api/v1/image", files=files, data=data)

    print("HTTP", resp.status_code)
    try:
        print(json.dumps(resp.json(), indent=2))
    except Exception:
        print("Raw text:", resp.text)

if __name__ == "__main__":
    run_test()
