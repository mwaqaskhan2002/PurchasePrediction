from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import pickle

app = FastAPI()
knn = pickle.load(open('model.pkl', 'rb'))

class PredictionRequest(BaseModel):
    age: int
    salary: int

@app.get('/')
def redirect_to_response():
    return RedirectResponse(url='/docs')

@app.post("/predict")
def predict(data: PredictionRequest):
    
    age = data.age
    salary = data.salary
    result = knn.predict([[age, salary]])[0]
    
    if result == 1:
        print('You are eligible to purchase')
    else:
        print('You are not eligible to purchase')

    return {"Prediction": {predict}, "result": int(result)}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='localhost', port=8000)

