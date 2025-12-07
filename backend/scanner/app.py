# app.py
import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path
from capture_and_infer import infer_from_image

SCANNER_DIR = Path(__file__).resolve().parent               # ...\backend\scanner
PROJECT_ROOT = SCANNER_DIR.parents[1]                       # ...\Project\Final Mehul
MODEL_PATH = PROJECT_ROOT.joinpath("models", "food_model.tflite")  # ...\models\food_model.tflite

app = Flask(__name__)
CORS(app)

# load ayurveda map
try:
    AYURVEDA = json.loads(SCANNER_DIR.joinpath("ayurveda_map.json").read_text(encoding="utf-8"))
except Exception:
    AYURVEDA = {}

@app.route("/scan", methods=["POST"])
def scan():
    """
    Accepts: multipart/form-data with 'image' file
    Returns: { label, confidence, nutrition, ayurveda }
    """
    if "image" not in request.files:
        return jsonify({"error": "No image provided. Use multipart form with 'image'."}), 400

    img_file = request.files["image"]
    save_path = SCANNER_DIR.joinpath("last_upload.jpg")
    img_file.save(str(save_path))

    # run inference
    label, confidence = infer_from_image(str(save_path), str(MODEL_PATH))

    # placeholder nutrition - integrate real API later
    nutrition = {
        "calories_per_100g": None,
        "serving_size": "100 g",
        "note": "Integrate Nutrition API (Nutritionix/Edamam) to fill this."
    }

    ayurveda = AYURVEDA.get(label.lower(), {"note": "No ayurveda data for this item."})

    return jsonify({
        "label": label,
        "confidence": confidence,
        "nutrition": nutrition,
        "ayurveda": ayurveda
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
