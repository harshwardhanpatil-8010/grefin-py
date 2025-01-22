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
    
    # Indian-specific benchmarks for missing values
    india_benchmarks = {
        "emissions": 200,  # Example: Max CO2 emissions per unit output (g/unit)
        "energy_consumption": 1000,  # Example: Max kWh/unit of production
        "waste": 50,  # Example: Waste in kg/unit
        "workforce_conditions": 80,  # Example: Workforce satisfaction score (0-100)
        "community_impact": 85,  # Example: Community benefit score (0-100)
        "compliance": 90,  # Example: Compliance percentage (0-100)
        "spend": 100,  # Example: Environmental spend per unit (in ₹)
        "activity": 100,  # Example: Activity level score (0-100)
    }

    # Fill missing values with Indian-specific benchmarks
    for feature, benchmark in india_benchmarks.items():
        input_data[feature].fillna(benchmark, inplace=True)

    # Normalize data using MinMaxScaler
    scaler = MinMaxScaler()
    input_data_scaled = scaler.fit_transform(input_data[features])

    return input_data_scaled, scaler

# Step 2: Define the AI Model (Random Forest Regressor)
def build_model():
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    return model

# Step 3: Load model
def load_model(filename='model.pkl'):
    return joblib.load(filename)

# Step 4: Create a Pydantic model for input
class IndustryData(BaseModel):
    emissions: float = None
    energy_consumption: float = None
    waste: float = None
    workforce_conditions: float = None
    community_impact: float = None
    compliance: float = None
    spend: float = None
    activity: float = None

# Step 5: Define industry-specific default values
industry_defaults = {
    "manufacturing": {
        "emissions": 180,
        "energy_consumption": 1500,
        "waste": 70,
        "workforce_conditions": 80,
        "community_impact": 85,
        "compliance": 90,
        "spend": 120,
        "activity": 100
    },
    "agriculture": {
        "emissions": 120,
        "energy_consumption": 900,
        "waste": 30,
        "workforce_conditions": 75,
        "community_impact": 80,
        "compliance": 85,
        "spend": 90,
        "activity": 90
    },
    "it_technology": {
    "emissions": 50,  # Low emissions due to virtual nature
    "energy_consumption": 600,  # Significant energy consumption from data centers
    "waste": 20,  # E-waste generation is minimal compared to manufacturing
    "workforce_conditions": 90,  # High focus on workforce satisfaction
    "community_impact": 85,  # Strong CSR initiatives
    "compliance": 95,  # High compliance with regulations
    "spend": 80,  # Moderate environmental spend
    "activity": 95  # High productivity level
},
    "construction": {
    "emissions": 250,  # High emissions from cement production and machinery
    "energy_consumption": 2000,  # Heavy equipment usage
    "waste": 120,  # Significant construction waste
    "workforce_conditions": 70,  # Challenges in maintaining workforce standards
    "community_impact": 75,  # Mixed impact on communities
    "compliance": 80,  # Adherence to building and environmental regulations
    "spend": 100,  # Moderate environmental spend
    "activity": 90  # High activity in the sector
},
    "healthcare": {
    "emissions": 100,  # Moderate emissions from operations
    "energy_consumption": 1200,  # Energy-intensive equipment
    "waste": 80,  # Biomedical waste is significant
    "workforce_conditions": 85,  # Generally good workforce standards
    "community_impact": 90,  # Positive community contributions
    "compliance": 90,  # Strict regulatory compliance
    "spend": 150,  # High spending on environmental initiatives
    "activity": 95  # High activity and demand
},
    "retail": {
    "emissions": 70,  # Moderate emissions from logistics
    "energy_consumption": 800,  # Store operations and cooling
    "waste": 40,  # Packaging and general waste
    "workforce_conditions": 80,  # Decent working conditions
    "community_impact": 85,  # CSR contributions vary
    "compliance": 90,  # Compliance with labor and environmental laws
    "spend": 75,  # Low to moderate environmental spending
    "activity": 85  # Regular activity levels
},
    "textile": {
    "emissions": 300,  # High emissions due to dyeing and chemical processes
    "energy_consumption": 1800,  # Energy-intensive machinery
    "waste": 100,  # High waste from fabric scraps and dyes
    "workforce_conditions": 65,  # Often poor workforce standards
    "community_impact": 70,  # Mixed community impacts
    "compliance": 75,  # Compliance is improving but inconsistent
    "spend": 90,  # Moderate spending on sustainable practices
    "activity": 90  # High production activity
},
    "automobile": {
    "emissions": 350,  # High emissions from manufacturing processes
    "energy_consumption": 2200,  # Energy-intensive operations
    "waste": 110,  # Significant waste from production lines
    "workforce_conditions": 75,  # Generally fair workforce standards
    "community_impact": 80,  # Moderate community contributions
    "compliance": 85,  # Strict regulations in the automobile sector
    "spend": 130,  # High spending on sustainability (e.g., EV development)
    "activity": 95  # High activity in production and sales
},
    "pharmaceuticals": {
    "emissions": 90,  # Moderate emissions from chemical processes
    "energy_consumption": 1400,  # Energy-intensive R&D and manufacturing
    "waste": 60,  # High focus on reducing medical and chemical waste
    "workforce_conditions": 85,  # Strong focus on workforce safety
    "community_impact": 90,  # Positive contributions through healthcare initiatives
    "compliance": 95,  # High compliance with regulations
    "spend": 120,  # High environmental and R&D spend
    "activity": 90  # Steady production activity
},
    "logistics": {
    "emissions": 400,  # High emissions from transportation
    "energy_consumption": 2000,  # Fuel and operational energy
    "waste": 50,  # Packaging and operational waste
    "workforce_conditions": 70,  # Variable workforce conditions
    "community_impact": 75,  # Limited community involvement
    "compliance": 85,  # Adherence to transport regulations
    "spend": 80,  # Low to moderate environmental spending
    "activity": 95  # High activity in supply chains
},

    # Add other industries as needed
}

# Step 6: Define API Endpoint for calculating the green score
@app.post("/get_green_score")
def get_green_score(data: IndustryData):
    try:
        # Convert the incoming data to a DataFrame
        input_data = pd.DataFrame([data.dict()])

        # Load the pre-trained model
        if os.path.exists("model.pkl"):
            model = load_model("model.pkl")
        else:
            raise HTTPException(status_code=404, detail="Model not found")

        # Preprocess input data (impute missing values and scale)
        input_data_scaled, _ = preprocess_data(input_data)

        # Predict the green score using the trained model
        green_score = model.predict(input_data_scaled)[0]

        # Return only the green score
        return {"green_score": green_score}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating Green Score: {str(e)}")

# Step 7: Define API Endpoint for industry-specific green score calculation
@app.post("/get_green_score/{industry}")
def get_green_score_for_industry(industry: str, data: IndustryData = None):
    try:
        # Check if industry defaults exist
        if industry not in industry_defaults:
            raise HTTPException(status_code=404, detail="Industry not found")

        # Use default values if no input data is provided
        industry_data = industry_defaults[industry]
        if data:
            # Override defaults with provided data
            provided_data = data.dict(exclude_unset=True)
            industry_data.update(provided_data)

        # Convert to DataFrame
        input_data = pd.DataFrame([industry_data])

        # Load the pre-trained model
        if os.path.exists("model.pkl"):
            model = load_model("model.pkl")
        else:
            raise HTTPException(status_code=404, detail="Model not found")

        # Preprocess input data (impute missing values and scale)
        input_data_scaled, _ = preprocess_data(input_data)

        # Predict the green score using the trained model
        green_score = model.predict(input_data_scaled)[0]

        # Return the green score
        return {"industry": industry, "green_score": green_score}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating Green Score: {str(e)}")

# Step 8: Training Utility (Run once to train and save the model)
def train_model():
    # Load Indian industry data (replace with your dataset)
    if not os.path.exists("india_industry_data.csv"):
        raise FileNotFoundError("Training data not found. Ensure 'india_industry_data.csv' is available.")

    data = pd.read_csv("india_industry_data.csv")
    
    # Features and target
    features = ["emissions", "energy_consumption", "waste", "workforce_conditions",
                "community_impact", "compliance", "spend", "activity"]
    target = "green_score"

    X = data[features]
    y = data[target]

    # Preprocess data
    X_scaled, scaler = preprocess_data(X)

    # Train the model
    model = build_model()
    model.fit(X_scaled, y)

    # Save the model and scaler
    joblib.dump(model, "model.pkl")
    joblib.dump(scaler, "scaler.pkl")
