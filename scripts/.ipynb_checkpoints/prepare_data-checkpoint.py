import os
from helpers.feast_utils import load_training_data, load_simulated_online_features

# Output directory
os.makedirs("data", exist_ok=True)

# Prepare training datasets for different year combinations
df_2023 = load_training_data(from_year=2023, to_year=2023)
df_2023.to_csv("data/train_2023.csv", index=False)

df_2024 = load_training_data(from_year=2024, to_year=2024)
df_2024.to_csv("data/train_2024.csv", index=False)

df_2023_2024 = load_training_data(from_year=2023, to_year=2024)
df_2023_2024.to_csv("data/train_2023_2024.csv", index=False)

# Prepare test dataset (simulated online features for 2025)
df_test_2025 = load_simulated_online_features(year=2025)
df_test_2025.to_csv("data/test_2025.csv", index=False)

print("Data preparation completed.")
