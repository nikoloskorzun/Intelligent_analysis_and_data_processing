from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and poly features
with open('/app/models/lab1/model_lab1.pkl', 'rb') as f:
    model = pickle.load(f)

with open('/app/models/lab1/poly_features_lab1.pkl', 'rb') as f:
    poly_features = pickle.load(f)

class Features(BaseModel):
    relative_compactness: float
    surface_area: float
    continuous_wall_area: float
    roof_area: float
    total_height: float
    orientation: float
    continuous_glazing_area: float
    glazing_area_distribution: float

@app.post("/predict")
def predict(features: Features):
    input_data = np.array([[
        features.relative_compactness,
        features.surface_area,
        features.continuous_wall_area,
        features.roof_area,
        features.total_height,
        features.orientation,
        features.continuous_glazing_area,
        features.glazing_area_distribution
    ]])
    
    # Transform features using polynomial transformer
    poly_data = poly_features.transform(input_data)
    prediction = model.predict(poly_data)
    
    return {"prediction": prediction.tolist()[0]}