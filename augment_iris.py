
import pandas as pd

# Load the dataset
df = pd.read_csv('data/iris.csv')

# Sample 20 additional rows for each species (with replacement)
additional_rows = df.groupby('species').apply(lambda x: x.sample(20, replace=True)).reset_index(drop=True)

# Concatenate the original and new rows
augmented_df = pd.concat([df, additional_rows])

# Save the new dataset
augmented_df.to_csv('data/iris.csv', index=False)

print('iris.csv dataset augmented with 60 more rows. Shape {augmented_df.shape}')
