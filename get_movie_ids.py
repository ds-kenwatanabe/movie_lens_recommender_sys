import argparse
import pandas as pd
from paths import DEFAULT_MOVIES_PATH


def parse_args():
    parser = argparse.ArgumentParser(description="Preview MovieLens movie IDs.")
    parser.add_argument("--movies-path", default=DEFAULT_MOVIES_PATH, help="Path to movies.csv.")
    parser.add_argument("--rows", type=int, default=5, help="Number of rows to display.")
    return parser.parse_args()


def main():
    args = parse_args()
    data = pd.read_csv(args.movies_path)
    print(data.head(args.rows))


if __name__ == "__main__":
    main()
