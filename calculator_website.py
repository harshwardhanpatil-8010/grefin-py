import numpy as np
import pandas as pd
import requests

class CarbonFootprintCalculator:
    def __init__(self, carbon_intensity=475, energy_intensity=0.81):
        self.carbon_intensity = carbon_intensity # grams CO2 per kWh
        self.energy_intensity = energy_intensity # kWh per GB

    def calculate_emissions_per_page_view(self, data_transfer_mb):
        data_transfer_gb = data_transfer_mb / 1024 # Convert MB to GB
        energy_consumed = data_transfer_gb * self.energy_intensity # kWh
        emissions = energy_consumed * self.carbon_intensity # grams CO2
        return emissions

    def calculate_annual_emissions(self, data_transfer_mb, annual_page_views):
        emissions_per_view = self.calculate_emissions_per_page_view(data_transfer_mb)
        annual_emissions = emissions_per_view * annual_page_views # grams CO2
        return annual_emissions

    def calculate_green_score(self, annual_emissions):
        green_score = max(0, 100 - (annual_emissions / 1000000)) # Example: Deduct 1 point per metric ton
        return green_score

if __name__ == "__main__":
    carbon_intensity = 475 # Global average in grams CO2 per kWh
    energy_intensity = 0.81 # kWh per GB of data transfer

    calculator = CarbonFootprintCalculator(carbon_intensity, energy_intensity)

    try:
        url = input("Enter the website URL: ")
        response = requests.get(url)
        data_transfer_mb = len(response.content) / (1024 ** 2) # Convert bytes to MB
        annual_page_views = int(input("Enter the estimated annual page views: "))

        annual_emissions = calculator.calculate_annual_emissions(data_transfer_mb, annual_page_views)
        green_score = calculator.calculate_green_score(annual_emissions)

        print(f"Annual CO2 Emissions for {url}: {annual_emissions / 1000:.2f} kg CO2")
        print(f"Green Score for {url}: {green_score}")
    except Exception as e:
        print(f"Error calculating emissions: {e}")