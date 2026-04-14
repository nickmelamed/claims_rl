import os
import json
from datetime import datetime


class ExperimentTracker:
    def __init__(self, exp_name="default"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_name = f"{exp_name}_{timestamp}"

        self.base_dir = os.path.join("artifacts", "experiments", self.exp_name)
        os.makedirs(self.base_dir, exist_ok=True)

        self.metrics = []

    def save_config(self, config: dict):
        path = os.path.join(self.base_dir, "config.json")
        with open(path, "w") as f:
            json.dump(config, f, indent=2)

    def log(self, metrics: dict):
        self.metrics.append(metrics)

        path = os.path.join(self.base_dir, "metrics.json")
        with open(path, "w") as f:
            json.dump(self.metrics, f, indent=2)

    def log_text(self, text: str):
        path = os.path.join(self.base_dir, "logs.txt")
        with open(path, "a") as f:
            f.write(text + "\n")