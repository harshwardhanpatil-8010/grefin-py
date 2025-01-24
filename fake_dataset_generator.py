import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_time_series_data(start_date='2020-01-01', periods=1000):
    # Create date range
    dates = pd.date_range(start=start_date, periods=periods, freq='D')
    
    industries = ['manufacturing', 'agriculture', 'it_technology', 'construction', 
                 'healthcare', 'retail', 'textile', 'automobile', 'pharmaceuticals', 'logistics']
    
    data_rows = []
    
    for industry in industries:
        for date in dates:
            # Base values specific to each industry
            base_values = {
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

            }
            
            # Seasonal factors (example: higher in summer)
            season_factor = 1 + 0.2 * np.sin(2 * np.pi * date.dayofyear / 365)
            
            # Generate correlated metrics
            emissions = base_values.get(industry, {'emissions': 50})['emissions'] * season_factor * (1 + np.random.normal(0, 0.1))
            energy_consumption = emissions * 10 * (1 + np.random.normal(0, 0.05))  # Correlation with emissions
            waste = emissions * 0.5 * (1 + np.random.normal(0, 0.08))  # Correlation with emissions
            
            # Workforce conditions (1-100 scale)
            safety_score = np.random.normal(85, 5)
            employee_satisfaction = np.random.normal(75, 8)
            
            # Compliance metrics
            compliance_score = np.random.normal(90, 5)
            violations = np.random.poisson(0.3)  # Random rare events
            
            # Financial metrics
            base_spend = {
                'manufacturing': 1000000,
                'agriculture': 500000,
                'it_technology': 800000,
                # Add other industries...
            }.get(industry, 700000)
            
            operational_spend = base_spend * (1 + np.random.normal(0, 0.1))
            
            row = {
                'date': date,
                'industry': industry,
                'co2_emissions': emissions,
                'energy_consumption_kwh': energy_consumption,
                'waste_tonnes': waste,
                'safety_score': safety_score,
                'employee_satisfaction': employee_satisfaction,
                'compliance_score': compliance_score,
                'violations': violations,
                'operational_spend': operational_spend,
                'season_factor': season_factor
            }
            
            data_rows.append(row)
    
    return pd.DataFrame(data_rows)

# Generate the dataset
df = generate_time_series_data(periods=365*3)  # 3 years of daily data

# Add some trends
df['energy_efficiency_trend'] = df.groupby('industry')['energy_consumption_kwh'].transform(
    lambda x: x * (1 - np.linspace(0, 0.2, len(x)))  # 20% improvement over time
)

# Export to different formats
df.to_csv('industry_sustainability_metrics.csv', index=False)
df.to_excel('industry_sustainability_metrics.xlsx', index=False)
df.to_json('industry_sustainability_metrics.json', orient='records')

# Print sample statistics
print("\nDataset Shape:", df.shape)
print("\nSample Statistics:")
print(df.describe())