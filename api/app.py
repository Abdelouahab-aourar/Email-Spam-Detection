from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

app = FastAPI(
    title="Email Spam Detection API",
    description="A simple API to classify emails as spam or ham using Logistic Regression model",
    version="1.0.0"
)

try:
    model = joblib.load('./ml/spam_model.pkl')
    vectorizer = joblib.load('./ml/vectorizer.pkl')
except FileNotFoundError:
    raise RuntimeError("Model or vectorizer file not found.")

class EmailRequest(BaseModel):
    message: str


@app.post("/predict")
def predict_spam(email: EmailRequest):
    try:
        input_features = vectorizer.transform([email.message])
        
        prediction = model.predict(input_features)

        label = "Ham" if prediction[0] == 0 else "Spam"
        
        return {
            "prediction": label,
            "spam_score": int(prediction[0])
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "Spam Detection API is running", "documentation": "For the built-in docs navigate to /docs or /redoc"}