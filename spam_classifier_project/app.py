from flask import Flask, request, render_template
import joblib

app = Flask(__name__)
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        input_text = request.form["sms"]
        vectorized = vectorizer.transform([input_text])
        result = model.predict(vectorized)[0]
        prediction = "SPAM" if result == 1 else "HAM (Not Spam)"
    return render_template("form.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)