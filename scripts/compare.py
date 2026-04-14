import os
import pandas as pd
import matplotlib.pyplot as plt


def compare_experiments(exp_paths):
    plt.figure()

    for path in exp_paths:
        df = pd.read_csv(os.path.join(path, "metrics.csv"))
        label = os.path.basename(path)

        plt.plot(df["episode"], df["reward"], label=label)

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Experiment Comparison")
    plt.legend()

    plt.savefig("artifacts/experiments/comparison.png")
    print("Saved comparison plot")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", nargs="+", required=True)
    args = parser.parse_args()

    compare_experiments(args.paths)