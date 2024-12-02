from fastapi import FastAPI
from pydantic import BaseModel
import pickle

# Initialize FastAPI app
app = FastAPI()

# Load the trained model
knn = pickle.load(open("model.pkl", "rb"))

# Define request body model
class PredictionRequest(BaseModel):
    age: int  
    salary: int 

# Define API endpoint for prediction
@app.post("/predict")
def predict(data: PredictionRequest):
    
    age = data.age
    salary = data.salary
    result = knn.predict([[age, salary]])[0]
    if result == 1:
        prediction = "Eligible to purchase" 
    else: 
        prediction ="Not eligible to purchase"
    
    return {"prediction": prediction, "result": int(result)}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run (app, host="localhost", port=8000)
