import numpy as np
import pandas as pd
import uvicorn
from pymongo import MongoClient
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import joblib
import datetime

# Load environment variables
load_dotenv()

# Step 1: Connect to MongoDB
client = MongoClient("mongodb+srv://harshwardhanpatil2005:LlnicvQxop7UTW07@grefin-web.ncahj.mongodb.net/?retryWrites=true&w=majority&appName=GREFIN-WEB")
db = client["industry_database"]  # Replace with your database name
collection = db["industry_data"]  # Replace with your collection name

# Step 2: Fetch Data from MongoDB
def fetch_data():
    data = list(collection.find())
    df = pd.DataFrame(data)
    print("Fetched data from MongoDB:")
    print(df.head())  # Log the first few records for inspection
    return df

# Step 3: Preprocess Data
def preprocess_data(df):
    df = df.drop(columns=['_id'], errors='ignore')  # Drop MongoDB-specific ID if present
    features = ["emissions", "energy_consumption", "waste", "workforce_conditions", 
                "community_impact", "compliance", "spend", "activity"]
    X = df[features]

    # Impute missing values using mean strategy
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # Scale data to range [0, 1]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    return X_scaled, scaler, imputer

# Step 4: Define the AI Model
def build_model():
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    return model

# Step 5: Train the Model
def train_model(X, y):
    model = build_model()
    model.fit(X, y)
    save_model(model)  # Save model after training
    return model

# Step 6: Save and Load Model
def save_model(model, filename='model.pkl'):
    joblib.dump(model, filename)

def load_model(filename='model.pkl'):
    return joblib.load(filename)

# Step 7: API Models for Input and Output
class IndustryData(BaseModel):
    industry_name: str
    emissions: float = None
    energy_consumption: float = None
    waste: float = None
    workforce_conditions: float = None
    community_impact: float = None
    compliance: float = None
    spend: float = None
    activity: float = None

class GreenScoreResponse(BaseModel):
    industry_name: str
    green_score: float
    message: str

# Step 8: FastAPI Application
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Green Score Calculator API"}

@app.post("/calculate_green_score", response_model=GreenScoreResponse)
def calculate_green_score_api(data: IndustryData):
    try:
        # Create DataFrame from input data
        input_data = pd.DataFrame([data.model_dump()])  # Updated for Pydantic v2

        # Fetch data from MongoDB
        df = fetch_data()

        # Ensure there is a green_score column in the database for training
        if "green_score" in df.columns and not df.empty:
            # Define features and target variable
            features = ["emissions", "energy_consumption", "waste", "workforce_conditions", 
                        "community_impact", "compliance", "spend", "activity"]
            target = "green_score"

            # Drop the '_id' field if present
            df = df.drop(columns=['_id'], errors='ignore')

            # Extract the features and target variables
            X = df[features]
            y = df[target]

            # Preprocess the data (imputation and scaling)
            X_scaled, scaler, imputer = preprocess_data(df)

            # Load or Train the Model
            if not os.path.exists("model.pkl"):
                model = train_model(X_scaled, y)  # Train and save the model if it doesn't exist
            else:
                model = load_model()  # Load the pre-trained model

            # Preprocess the input data
            input_data_imputed = imputer.transform(input_data.drop(columns=["industry_name"], errors='ignore'))
            input_data_scaled = scaler.transform(input_data_imputed)

            # Predict the green score for the new industry
            green_score = model.predict(input_data_scaled)[0]

            # Return the response with the calculated green score
            response = GreenScoreResponse(
                industry_name=data.industry_name,
                green_score=green_score,
                message="Green Score calculated successfully."
            )
            return response

        else:
            # Use heuristic estimation if no green_score column is found or not enough data
            green_score = estimate_green_score_for_new_industry(input_data)

            response = GreenScoreResponse(
                industry_name=data.industry_name,
                green_score=green_score,
                message="Green Score calculated using heuristic."
            )
            return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating Green Score: {str(e)}")


# Placeholder function for heuristic estimation
def estimate_green_score_for_new_industry(input_data):
    return np.random.uniform(50, 100)  # Example heuristic: return a random score

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)