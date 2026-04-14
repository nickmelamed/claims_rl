import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_experiment(exp_path):
    csv_path = os.path.join(exp_path, "metrics.csv")

    df = pd.read_csv(csv_path)

    plt.figure()
    plt.plot(df["episode"], df["reward"])
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward Curve")

    out_path = os.path.join(exp_path, "reward_curve.png")
    plt.savefig(out_path)

    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    args = parser.parse_args()

    plot_experiment(args.path)