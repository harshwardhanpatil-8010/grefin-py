from pymongo import MongoClient
import pandas as pd

# Connect to MongoDB
client = MongoClient("mongodb+srv://harshwardhanpatil2005:LlnicvQxop7UTW07@grefin-web.ncahj.mongodb.net/?retryWrites=true&w=majority&appName=GREFIN-WEB")
db = client["industry_database"]
collection = db["industry_data"]

# Example industry data
data = [
    {
        "industry": "manufacturing",
        "co2_emissions": 156.96,
        "energy_consumption_kwh": 1353.91,
        "waste_tonnes": 86.51,
        "safety_score": 83.37,
        "employee_satisfaction": 59.47,
        "compliance_score": 90.27,
        "violations": 0,
        "operational_spend": 955037.85,
        "season_factor": 1.003,
        "energy_efficiency_trend": 1353.91,
    },
    {
        "industry": "agriculture",
        "co2_emissions": 120.54,
        "energy_consumption_kwh": 900.34,
        "waste_tonnes": 30.12,
        "safety_score": 75.50,
        "employee_satisfaction": 80.30,
        "compliance_score": 85.60,
        "violations": 1,
        "operational_spend": 500000.00,
        "season_factor": 1.01,
        "energy_efficiency_trend": 900.10,
    },
     {
        "industry": "it_technology",
        "co2_emissions": 50.34,
        "energy_consumption_kwh": 1200.23,
        "waste_tonnes": 15.25,
        "safety_score": 90.50,
        "employee_satisfaction": 88.30,
        "compliance_score": 95.10,
        "violations": 0,
        "operational_spend": 600000.00,
        "season_factor": 1.02,
        "energy_efficiency_trend": 1200.00
    },
    {
        "industry": "construction",
        "co2_emissions": 250.67,
        "energy_consumption_kwh": 2000.89,
        "waste_tonnes": 120.45,
        "safety_score": 70.40,
        "employee_satisfaction": 65.80,
        "compliance_score": 85.20,
        "violations": 2,
        "operational_spend": 850000.00,
        "season_factor": 1.03,
        "energy_efficiency_trend": 2001.50
    },
    {
        "industry": "healthcare",
        "co2_emissions": 100.89,
        "energy_consumption_kwh": 1500.45,
        "waste_tonnes": 80.50,
        "safety_score": 85.70,
        "employee_satisfaction": 84.60,
        "compliance_score": 90.50,
        "violations": 1,
        "operational_spend": 750000.00,
        "season_factor": 1.01,
        "energy_efficiency_trend": 1499.50
    },
    {
        "industry": "retail",
        "co2_emissions": 70.12,
        "energy_consumption_kwh": 950.30,
        "waste_tonnes": 35.20,
        "safety_score": 78.90,
        "employee_satisfaction": 80.10,
        "compliance_score": 89.30,
        "violations": 0,
        "operational_spend": 400000.00,
        "season_factor": 1.04,
        "energy_efficiency_trend": 945.60
    },
    {
        "industry": "textile",
        "co2_emissions": 300.45,
        "energy_consumption_kwh": 1800.90,
        "waste_tonnes": 105.30,
        "safety_score": 65.40,
        "employee_satisfaction": 58.20,
        "compliance_score": 78.60,
        "violations": 3,
        "operational_spend": 500000.00,
        "season_factor": 1.02,
        "energy_efficiency_trend": 1801.00
    },
    {
        "industry": "automobile",
        "co2_emissions": 350.23,
        "energy_consumption_kwh": 2100.50,
        "waste_tonnes": 115.45,
        "safety_score": 75.60,
        "employee_satisfaction": 70.80,
        "compliance_score": 82.40,
        "violations": 2,
        "operational_spend": 900000.00,
        "season_factor": 1.05,
        "energy_efficiency_trend": 2099.80
    },
    {
        "industry": "pharmaceuticals",
        "co2_emissions": 90.80,
        "energy_consumption_kwh": 1400.50,
        "waste_tonnes": 65.40,
        "safety_score": 87.20,
        "employee_satisfaction": 83.50,
        "compliance_score": 92.80,
        "violations": 0,
        "operational_spend": 850000.00,
        "season_factor": 1.02,
        "energy_efficiency_trend": 1401.20
    },
    {
        "industry": "logistics",
        "co2_emissions": 400.45,
        "energy_consumption_kwh": 2500.30,
        "waste_tonnes": 50.20,
        "safety_score": 68.40,
        "employee_satisfaction": 60.50,
        "compliance_score": 80.20,
        "violations": 5,
        "operational_spend": 700000.00,
        "season_factor": 1.03,
        "energy_efficiency_trend": 2499.50
    }
]

# Insert data into MongoDB
collection.insert_many(data)
print("Data inserted successfully!")
