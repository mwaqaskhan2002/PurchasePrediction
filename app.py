from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import  HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import RedirectResponse
from dotenv import load_dotenv
import os
import pickle

load_dotenv()
app = FastAPI()
security = HTTPBearer()
knn = pickle.load(open("model.pkl", "rb"))

API_KEY = os.getenv("API_KEY")

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )


@app.get("/")
def read_root():
    return RedirectResponse(url="/docs")

@app.post("/predict")
def predict(age: int, salary: int, credentials: HTTPAuthorizationCredentials = Depends(security)):
    
    result = knn.predict([[age, salary]])[0]
    if result == 1:
        prediction = "Eligible to purchase" 
    else: 
        prediction ="Not eligible to purchase"
    
    return {"prediction": prediction, "result": int(result)}


