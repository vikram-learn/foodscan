# capture_and_infer.py
import cv2
import numpy as np
from pathlib import Path

# Try to import tensorflow for TFLite runtime
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

CURRENT_DIR = Path(__file__).resolve().parent
LABELS_FILE = CURRENT_DIR.joinpath("labels.txt")

def load_labels():
    if LABELS_FILE.exists():
        return [l.strip() for l in LABELS_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]
    return ["banana", "apple", "pizza", "fried_rice", "samosa"]

def infer_with_tflite(model_path, image_path):
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow not available for TFLite inference.")
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # basic preprocessing: resize to model input shape, normalize
    inp_shape = input_details[0]['shape']  # [1, H, W, C]
    H, W = int(inp_shape[1]), int(inp_shape[2])

    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (W, H))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]['index'])
    probs = np.squeeze(out)
    top_idx = int(np.argmax(probs))
    confidence = float(probs[top_idx])
    labels = load_labels()
    label = labels[top_idx] if top_idx < len(labels) else f"class_{top_idx}"
    return label, confidence

def stub_infer(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        return "unknown", 0.0
    avg = img.mean(axis=(0,1))  # BGR
    b,g,r = avg
    # naive heuristic purely for demo
    if r > 120 and g > 100 and b > 60:
        return "banana", 0.85
    if r > 100 and g < 100 and b < 100:
        return "pizza", 0.60
    return "apple", 0.70

def infer_from_image(image_path, model_path):
    model_p = Path(model_path)
    if model_p.exists() and TF_AVAILABLE:
        try:
            return infer_with_tflite(model_p, image_path)
        except Exception as e:
            print("TFLite inference failed:", e)
            print("Falling back to stub inference.")
            return stub_infer(image_path)
    else:
        return stub_infer(image_path)
