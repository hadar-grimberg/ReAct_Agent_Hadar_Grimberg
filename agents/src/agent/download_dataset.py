#!/usr/bin/env python3
"""
download_dataset.py — Download the real Bitext customer-support dataset from HuggingFace.

The full dataset has ~27,000 rows across 27 intents and 11 categories.
Requires: pip install datasets

Usage:
    python download_dataset.py
    python download_dataset.py --output my_dataset.csv
"""
import argparse
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="bitext_full.csv",
                        help="Output CSV path (default: bitext_full.csv)")
    parser.add_argument("--split", default="train",
                        help="Dataset split to download (default: train)")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package not installed.")
        print("Run: pip install datasets")
        sys.exit(1)

    print("Downloading Bitext dataset from HuggingFace...")
    print("Dataset: bitext/Bitext-customer-support-llm-chatbot-training-dataset")

    try:
        ds = load_dataset(
            "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
            split=args.split,
        )
        df = ds.to_pandas()
        df.to_csv(args.output, index=False)
        print(f"✓ Saved {len(df):,} rows to {args.output}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Categories: {sorted(df['category'].unique().tolist())}")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
