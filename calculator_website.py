import numpy as np
import pandas as pd
import requests

class CarbonFootprintCalculator:
    def __init__(self, electricity_emission_factor, server_power_usage, energy_intensity_of_network, device_power_usage):
        self.electricity_emission_factor = electricity_emission_factor
        self.server_power_usage = server_power_usage
        self.energy_intensity_of_network = energy_intensity_of_network
        self.device_power_usage = device_power_usage

    def calculate_server_emissions(self, hours):
        return self.server_power_usage * hours * self.electricity_emission_factor

    def calculate_network_emissions(self, data_transferred):
        return data_transferred * self.energy_intensity_of_network

    def calculate_device_emissions(self, hours):
        return self.device_power_usage * hours * self.electricity_emission_factor

    def calculate_total_emissions(self, hours, data_transferred):
        server_emissions = self.calculate_server_emissions(hours)
        network_emissions = self.calculate_network_emissions(data_transferred)
        device_emissions = self.calculate_device_emissions(hours)
        return server_emissions + network_emissions + device_emissions

    def calculate_green_score(self, url):
        try:
            response = requests.get(url)
            data_transferred = len(response.content) / (1024 ** 2) # Convert bytes to megabytes
            hours = 0.1 # Assuming 6 minutes of usage
            total_emissions = self.calculate_total_emissions(hours, data_transferred)
            green_score = max(0, 100 - total_emissions) 
            return green_score
        except Exception as e:
            print(f"Error calculating green score: {e}")
            return None

if __name__ == "__main__":
    electricity_emission_factor = 0.5
    server_power_usage = 300
    energy_intensity_of_network = 0.02
    device_power_usage = 50

    calculator = CarbonFootprintCalculator(
        electricity_emission_factor, 
        server_power_usage, 
        energy_intensity_of_network, 
        device_power_usage
    )

    url = input("Enter the website URL: ")
    green_score = calculator.calculate_green_score(url)
    if green_score is not None:
        print(f"Green Score for {url}: {green_score}")
