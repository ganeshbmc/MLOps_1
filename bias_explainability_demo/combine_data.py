import pandas as pd

# File paths
file1 = "train_2023_2024.csv"
file2 = "test_2025.csv"

# Read both CSVs
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# Reorder columns of df2 to match df1
df2 = df2[df1.columns]

# Combine rows
combined_df = pd.concat([df1, df2], ignore_index=True)

# Save the combined CSV
combined_df.to_csv("iris_combined.csv", index=False)

print(f"Combined CSV saved to iris_combined.csv")
