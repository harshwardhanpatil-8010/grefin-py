import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import joblib

# FastAPI application
app = FastAPI()

# Step 1: Preprocess the input data
def preprocess_data(input_data):
    # Features for which the score is calculated
    features = ["emissions", "energy_consumption", "waste", "workforce_conditions",
                "community_impact", "compliance", "spend", "activity"]
    # We need to impute missing values and scale the data
    imputer = SimpleImputer(strategy='mean')
    scaler = MinMaxScaler()

    # Impute and scale the input data
    input_data_imputed = imputer.fit_transform(input_data[features])
    input_data_scaled = scaler.fit_transform(input_data_imputed)

    return input_data_scaled, scaler, imputer

# Step 2: Define the AI Model (Random Forest Regressor)
def build_model():
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    return model

# Step 3: Load model
def load_model(filename='model.pkl'):
    return joblib.load(filename)

# Step 4: Create a Pydantic model for input
class IndustryData(BaseModel):
    emissions: float
    energy_consumption: float
    waste: float
    workforce_conditions: float
    community_impact: float
    compliance: float
    spend: float
    activity: float

# Step 5: Define API Endpoint for calculating the green score
@app.post("/get_green_score")
def get_green_score(data: IndustryData):
    try:
        # Convert the incoming data to a DataFrame
        input_data = pd.DataFrame([data.dict()])

        # Load or Train Model
        if os.path.exists("model.pkl"):
            model = load_model()  # Load pre-trained model
        else:
            raise HTTPException(status_code=404, detail="Model not found")

        # Preprocess input data (impute missing values and scale)
        input_data_scaled, scaler, imputer = preprocess_data(input_data)

        # Predict the green score using the trained model
        green_score = model.predict(input_data_scaled)[0]

        # Return only the green score
        return {"green_score": green_score}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating Green Score: {str(e)}")
