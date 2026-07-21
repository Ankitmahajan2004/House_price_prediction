import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# Load the pickle model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'Liner_model_pkl')

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

# Exact feature list extracted from your scikit-learn model
FEATURES = [
    {"id": "id", "label": "ID", "default": 1, "step": 1, "type": "int"},
    {"id": "Date", "label": "Date (e.g. 20140502)", "default": 20140502, "step": 1, "type": "int"},
    {"id": "number of bedrooms", "label": "Bedrooms", "default": 3, "step": 1, "type": "int"},
    {"id": "number of bathrooms", "label": "Bathrooms", "default": 2.0, "step": 0.25, "type": "float"},
    {"id": "living area", "label": "Living Area (sqft)", "default": 2000, "step": 10, "type": "float"},
    {"id": "lot area", "label": "Lot Area (sqft)", "default": 5000, "step": 10, "type": "float"},
    {"id": "number of floors", "label": "Floors", "default": 1.5, "step": 0.5, "type": "float"},
    {"id": "waterfront present", "label": "Waterfront Present (0/1)", "default": 0, "step": 1, "type": "int"},
    {"id": "number of views", "label": "Number of Views (0-4)", "default": 0, "step": 1, "type": "int"},
    {"id": "condition of the house", "label": "House Condition (1-5)", "default": 3, "step": 1, "type": "int"},
    {"id": "grade of the house", "label": "House Grade (1-13)", "default": 7, "step": 1, "type": "int"},
    {"id": "Area of the house(excluding basement)", "label": "Area Above Basement (sqft)", "default": 1500, "step": 10, "type": "float"},
    {"id": "Area of the basement", "label": "Basement Area (sqft)", "default": 500, "step": 10, "type": "float"},
    {"id": "Built Year", "label": "Built Year", "default": 1990, "step": 1, "type": "int"},
    {"id": "Renovation Year", "label": "Renovation Year (0 if none)", "default": 0, "step": 1, "type": "int"},
    {"id": "Postal Code", "label": "Postal Code", "default": 98001, "step": 1, "type": "int"},
    {"id": "Lattitude", "label": "Latitude", "default": 47.5, "step": 0.0001, "type": "float"},
    {"id": "Longitude", "label": "Longitude", "default": -122.2, "step": 0.0001, "type": "float"},
    {"id": "living_area_renov", "label": "Living Area Renovated", "default": 1800, "step": 10, "type": "float"},
    {"id": "lot_area_renov", "label": "Lot Area Renovated", "default": 5000, "step": 10, "type": "float"},
    {"id": "Number of schools nearby", "label": "Nearby Schools", "default": 2, "step": 1, "type": "int"},
    {"id": "Distance from the airport", "label": "Distance from Airport (miles)", "default": 15, "step": 1, "type": "float"}
]

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>House Price Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
            --card-bg: rgba(30, 41, 59, 0.7);
            --accent-glow: #6366f1;
            --accent-color: #818cf8;
            --text-main: #f8fafc;
            --text-sub: #94a3b8;
            --input-bg: rgba(15, 23, 42, 0.6);
            --border-color: rgba(255, 255, 255, 0.1);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        body {
            background: var(--bg-gradient);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 2rem 1rem;
            overflow-x: hidden;
        }

        /* Ambient Background Animations */
        .glow-circle {
            position: absolute;
            width: 400px;
            height: 400px;
            background: radial-gradient(circle, rgba(99, 102, 241, 0.25) 0%, rgba(0, 0, 0, 0) 70%);
            border-radius: 50%;
            z-index: 0;
            animation: float 8s ease-in-out infinite alternate;
        }

        .glow-1 { top: -100px; left: -100px; }
        .glow-2 { bottom: -100px; right: -100px; animation-delay: -4s; }

        @keyframes float {
            0% { transform: translate(0, 0) scale(1); }
            100% { transform: translate(30px, 50px) scale(1.1); }
        }

        .container {
            position: relative;
            z-index: 1;
            width: 100%;
            max-width: 1100px;
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--border-color);
            border-radius: 24px;
            padding: 2.5rem;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            animation: fadeIn 0.8s ease-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        header {
            text-align: center;
            margin-bottom: 2rem;
        }

        header h1 {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(to right, #ffffff, var(--accent-color));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }

        header p {
            color: var(--text-sub);
            font-size: 1rem;
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
            gap: 1.25rem;
            margin-bottom: 2rem;
        }

        .input-group {
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }

        .input-group label {
            font-size: 0.85rem;
            color: var(--text-sub);
            font-weight: 600;
        }

        .input-group input {
            background: var(--input-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 0.75rem 1rem;
            color: var(--text-main);
            font-size: 0.95rem;
            outline: none;
            transition: all 0.3s ease;
        }

        .input-group input:focus {
            border-color: var(--accent-glow);
            box-shadow: 0 0 12px rgba(99, 102, 241, 0.4);
            transform: translateY(-2px);
        }

        .btn-submit {
            grid-column: 1 / -1;
            background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 1rem;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 10px 20px -5px rgba(99, 102, 241, 0.4);
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 0.5rem;
        }

        .btn-submit:hover {
            transform: translateY(-3px);
            box-shadow: 0 15px 25px -5px rgba(99, 102, 241, 0.6);
        }

        .btn-submit:active {
            transform: translateY(-1px);
        }

        /* Output Box Styles */
        .result-container {
            margin-top: 1.5rem;
            padding: 1.5rem;
            border-radius: 16px;
            background: rgba(15, 23, 42, 0.8);
            border: 1px solid var(--border-color);
            text-align: center;
            display: none;
            animation: slideUp 0.5s cubic-bezier(0.16, 1, 0.3, 1);
        }

        @keyframes slideUp {
            from { opacity: 0; transform: translateY(15px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .result-container h3 {
            color: var(--text-sub);
            font-size: 1rem;
            margin-bottom: 0.5rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .price-output {
            font-size: 3rem;
            font-weight: 700;
            color: #34d399;
            text-shadow: 0 0 20px rgba(52, 211, 153, 0.3);
        }

        /* Spinner */
        .spinner {
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top-color: white;
            animation: spin 0.8s linear infinite;
            display: none;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="glow-circle glow-1"></div>
    <div class="glow-circle glow-2"></div>

    <div class="container">
        <header>
            <h1>House Valuation AI</h1>
            <p>Enter property specifications below for instant price estimation</p>
        </header>

        <form id="predict-form">
            <div class="form-grid">
                {% for feat in features %}
                <div class="input-group">
                    <label for="{{ feat.id }}">{{ feat.label }}</label>
                    <input 
                        type="number" 
                        id="{{ feat.id }}" 
                        name="{{ feat.id }}" 
                        value="{{ feat.default }}" 
                        step="{{ feat.step }}" 
                        required>
                </div>
                {% endfor %}
            </div>

            <button type="submit" class="btn-submit" id="submit-btn">
                <span id="btn-text">Estimate Property Value</span>
                <div class="spinner" id="spinner"></div>
            </button>
        </form>

        <div class="result-container" id="result-box">
            <h3>Estimated Market Value</h3>
            <div class="price-output" id="price-display">$0</div>
        </div>
    </div>

    <script>
        document.getElementById('predict-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const submitBtn = document.getElementById('submit-btn');
            const btnText = document.getElementById('btn-text');
            const spinner = document.getElementById('spinner');
            const resultBox = document.getElementById('result-box');
            const priceDisplay = document.getElementById('price-display');

            // UI Feedback state
            btnText.textContent = "Calculating...";
            spinner.style.display = "block";
            submitBtn.style.opacity = "0.8";
            submitBtn.disabled = true;

            const formData = new FormData(e.target);
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();

                if (data.status === 'success') {
                    resultBox.style.display = 'block';
                    
                    // Animated Counter Effect
                    let startVal = 0;
                    const endVal = data.prediction;
                    const duration = 1000;
                    const startTime = performance.now();

                    function updateCounter(currentTime) {
                        const elapsed = currentTime - startTime;
                        const progress = Math.min(elapsed / duration, 1);
                        
                        // Ease out cubic function
                        const easeProgress = 1 - Math.pow(1 - progress, 3);
                        const currentVal = startVal + (endVal - startVal) * easeProgress;
                        
                        priceDisplay.textContent = '$' + currentVal.toLocaleString('en-US', {
                            minimumFractionDigits: 2,
                            maximumFractionDigits: 2
                        });

                        if (progress < 1) {
                            requestAnimationFrame(updateCounter);
                        }
                    }

                    requestAnimationFrame(updateCounter);
                } else {
                    alert('Error making prediction: ' + data.error);
                }
            } catch (err) {
                alert('An error occurred connecting to the server.');
            } finally {
                // Reset button state
                btnText.textContent = "Estimate Property Value";
                spinner.style.display = "none";
                submitBtn.style.opacity = "1";
                submitBtn.disabled = false;
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, features=FEATURES)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'status': 'error', 'error': 'Model file not loaded.'}), 500

    try:
        # Construct feature array strictly following trained model order
        feature_values = []
        for feat in FEATURES:
            val = request.form.get(feat['id'])
            feature_values.append(float(val))

        # Convert to 2D NumPy array
        input_array = np.array([feature_values])

        # Model Prediction
        prediction = model.predict(input_array)[0]

        return jsonify({
            'status': 'success',
            'prediction': float(prediction)
        })
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
