import argparse
import pandas as pd
import numpy as np

def poison_features(input_path, output_path, columns, noise_ratio, noise_std, seed):
    np.random.seed(seed)

    df = pd.read_parquet(input_path)
    n = len(df)
    poison_idx = np.random.choice(n, int(n * noise_ratio), replace=False)

    for col in columns:
        if col in df.columns:
            noise = np.random.normal(loc=0, scale=noise_std, size=len(poison_idx))
            df.loc[poison_idx, col] += noise
        else:
            print(f"Warning: Column '{col}' not found in {input_path}.")

    df.to_parquet(output_path, index=False)
    print(f"Poisoned features written to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add Gaussian noise to selected features.")
    parser.add_argument("--input", required=True, help="Input Parquet file path.")
    parser.add_argument("--output", required=True, help="Output Parquet file path.")
    parser.add_argument("--columns", nargs='+', required=True, help="List of feature columns to poison.")
    parser.add_argument("--noise_ratio", type=float, default=0.1, help="Fraction of rows to poison.")
    parser.add_argument("--noise_std", type=float, default=0.5, help="Standard deviation of noise.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    args = parser.parse_args()

    poison_features(
        input_path=args.input,
        output_path=args.output,
        columns=args.columns,
        noise_ratio=args.noise_ratio,
        noise_std=args.noise_std,
        seed=args.seed
    )
