import argparse
import pandas as pd
import numpy as np
import random

def poison_labels(input_path, output_path, label_col, flip_ratio, seed):
    np.random.seed(seed)
    random.seed(seed)

    df = pd.read_csv(input_path)
    n = len(df)

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in {input_path}.")

    unique_labels = df[label_col].unique().tolist()
    poison_idx = np.random.choice(n, int(n * flip_ratio), replace=False)

    def flip(label):
        choices = [l for l in unique_labels if l != label]
        return random.choice(choices)

    df.loc[poison_idx, label_col] = df.loc[poison_idx, label_col].apply(flip)
    df.to_csv(output_path, index=False)
    print(f"Flipped labels written to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flip labels randomly.")
    parser.add_argument("--input", required=True, help="Input CSV file path.")
    parser.add_argument("--output", required=True, help="Output CSV file path.")
    parser.add_argument("--label_col", required=True, help="Name of the label column to flip.")
    parser.add_argument("--flip_ratio", type=float, default=0.1, help="Fraction of labels to flip.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    args = parser.parse_args()

    poison_labels(
        input_path=args.input,
        output_path=args.output,
        label_col=args.label_col,
        flip_ratio=args.flip_ratio,
        seed=args.seed
    )
