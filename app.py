from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__, template_folder="templates")
CORS(app)

MODEL = joblib.load('model.pkl')
SCALER = joblib.load('scaler.pkl')
ENCODERS = joblib.load('encoders.pkl')
try:
    Y_ENCODER = joblib.load('y_encoder.pkl')
except Exception:
    Y_ENCODER = None

FEATURE_ORDER = [
    'age','workclass','fnlwgt','education-num','marital-status','occupation',
    'relationship','race','gender','capital-gain','capital-loss','hours-per-week','native-country'
]

def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)

def _encode_value(col, raw_val):
    enc = ENCODERS.get(col) if isinstance(ENCODERS, dict) else None
    if enc is None:
        return _safe_float(raw_val, 0.0)

    s = str(raw_val).strip()
    try:
        return int(enc.transform([s])[0])
    except Exception:
        pass

    try:
        classes = list(getattr(enc, 'classes_', []))
        for c in classes:
            if str(c).lower() == s.lower():
                return int(enc.transform([c])[0])
    except Exception:
        pass

    try:
        return int(enc.transform([classes[0]])[0])
    except Exception:
        return 0

@app.route('/')
def index():
    return render_template('employ.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True) or {}
        x = [_encode_value(col, data.get(col, 0)) for col in FEATURE_ORDER]
        x = np.asarray(x, dtype=float).reshape(1, -1)

        x_scaled = SCALER.transform(x)
        raw_pred = MODEL.predict(x_scaled)[0]
        pred_round = int(round(float(raw_pred)))

        if Y_ENCODER is not None:
            try:
                label = str(Y_ENCODER.inverse_transform([pred_round])[0])
            except Exception:
                label = ">50K" if pred_round == 1 else "<=50K"
        else:
            label = ">50K" if pred_round == 1 else "<=50K"

        # 🔹 Only return label (not raw number)
        return jsonify({"label": label}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
