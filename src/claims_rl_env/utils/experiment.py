import os
import json
import csv
from datetime import datetime


class ExperimentTracker:
    def __init__(self, exp_name="default"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_name = f"{exp_name}_{timestamp}"

        self.base_dir = os.path.join("artifacts", "experiments", self.exp_name)
        os.makedirs(self.base_dir, exist_ok=True)

        self.metrics = []
        self.csv_path = os.path.join(self.base_dir, "metrics.csv")

    def save_config(self, config: dict):
        with open(os.path.join(self.base_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    def log(self, metrics: dict):
        self.metrics.append(metrics)

        # JSON
        with open(os.path.join(self.base_dir, "metrics.json"), "w") as f:
            json.dump(self.metrics, f, indent=2)

        # CSV (append)
        write_header = not os.path.exists(self.csv_path)
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metrics.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(metrics)

    def get_dir(self):
        return self.base_dir