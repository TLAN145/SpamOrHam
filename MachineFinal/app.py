from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import pickle

app = FastAPI()

# Load trained model
with open("spam_classifier.pkl", "rb") as f:
    model = pickle.load(f)

# Home page with centered input form
@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <html>
        <head>
            <title>Spam Classifier</title>
            <style>
                html, body {
                    height: 100%;
                    margin: 0;
                    font-family: Arial, sans-serif;
                    background-color: #f0f4f8;
                }
                .center-container {
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    height: 100%;
                    flex-direction: column;
                }
                form {
                    background-color: #ffffff;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                    width: 90%;
                    max-width: 600px;
                }
                textarea {
                    width: 100%;
                    padding: 10px;
                    font-size: 16px;
                    border-radius: 5px;
                    border: 1px solid #ccc;
                }
                input[type="submit"] {
                    background-color: #4CAF50;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 5px;
                    cursor: pointer;
                    font-size: 16px;
                    margin-top: 10px;
                }
                input[type="submit"]:hover {
                    background-color: #45a049;
                }
                h2 {
                    color: #333;
                    margin-bottom: 20px;
                }
            </style>
        </head>
        <body>
            <div class="center-container">
                <h2>Enter email text to classify:</h2>
                <form action="/predict" method="post">
                    <textarea name="text" rows="10" required></textarea><br><br>
                    <input type="submit" value="Classify">
                </form>
            </div>
        </body>
    </html>
    """

# Prediction endpoint with centered result
@app.post("/predict", response_class=HTMLResponse)
async def predict(text: str = Form(...)):
    prediction = model.predict([text])[0]
    result = "Spam" if prediction == 1 else "Not Spam"
    color = "#e74c3c" if result == "Spam" else "#2ecc71"

    return f"""
    <html>
        <head>
            <style>
                html, body {{
                    height: 100%;
                    margin: 0;
                    font-family: Arial, sans-serif;
                    background-color: #f0f4f8;
                }}
                .center-container {{
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                    align-items: center;
                    height: 100%;
                    text-align: center;
                }}
                .result {{
                    font-size: 28px;
                    font-weight: bold;
                    color: {color};
                    margin-bottom: 20px;
                }}
                a {{
                    font-size: 18px;
                    color: #3498db;
                    text-decoration: none;
                }}
                a:hover {{
                    text-decoration: underline;
                }}
            </style>
        </head>
        <body>
            <div class="center-container">
                <h2>Prediction Result:</h2>
                <div class="result">🧠 {result}</div>
                <a href="/">🔁 Try another</a>
            </div>
        </body>
    </html>
    """
