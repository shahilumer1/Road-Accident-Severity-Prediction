from flask import Flask, render_template, request
import pickle
import numpy as np

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load mappings
with open("mappings.pkl", "rb") as f:
    mappings = pickle.load(f)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    # Load dropdown options from mappings
    options = mappings["options"]

    if request.method == "POST":
        try:
            weather = request.form["weather"]
            road = request.form["road"]
            vehicle = request.form["vehicle"]
            light = request.form["light"]

            # Encode using mappings
            w = mappings["weather"][weather]
            r = mappings["road"][road]
            v = mappings["vehicle"][vehicle]
            l = mappings["light"][light]

            # Prepare input
            X = np.array([[w, r, v, l]])

            # Predict
            pred = model.predict(X)[0]
            prediction = mappings["severity_rev"][pred]

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template(
        "index.html",
        options=options,
        prediction=prediction
    )

if __name__ == "__main__":
    app.run(debug=True)
