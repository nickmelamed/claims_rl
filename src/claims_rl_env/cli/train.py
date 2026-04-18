from claims_rl_env.environment.environment import ClaimEnv
from claims_rl_env.environment.curriculum import Curriculum
from claims_rl_env.agent.policy import SoftmaxPolicy
from claims_rl_env.environment.actions import ACTIONS
from claims_rl_env.data.dataset import load_dataset
from claims_rl_env.utils.experiment import ExperimentTracker

import argparse
import random
import numpy as np


def train(episodes, method="ppo"):
    dataset = load_dataset()
    curriculum = Curriculum()

    sampled_dataset = [curriculum.sample(dataset) for _ in range(len(dataset))]
    env = ClaimEnv(sampled_dataset)

    n_actions = len(list(ACTIONS))
    policy = SoftmaxPolicy(n_actions)

    tracker = ExperimentTracker(exp_name=f"{method}_run")

    # save config
    tracker.save_config({
        "rl_method": method,
        "num_episodes": episodes,
        "dataset_size": len(dataset),
        "curriculum": True,
    })

    # training loop
    for ep in range(episodes):
        state = env.reset()
        done = False

        total_reward = 0
        step = 0

        trajectory = []

        support_count = 0
        contradict_count = 0
        removed_count = 0

        while not done:
            action, payload = policy.act(state)

            next_state, reward, done, _ = env.step(action, payload)

            total_reward += reward
            step += 1

            # track behavior
            if action == "generate_support_argument":
                support_count += 1
            elif action == "generate_contradict_argument":
                contradict_count += 1
            elif action == "remove_evidence":
                removed_count += 1

            trajectory.append({
                "step": step,
                "action": action,
                "reward": reward,
                "selected_ids": [e.id for e in next_state.selected_evidence],
            })

            state = next_state

        # episode logging
        tracker.log_episode({
            "episode": ep,
            "reward": total_reward,
            "num_steps": step,
            "final_decision": getattr(state, "final_decision", None),
            "correct": getattr(state, "correct", None),
            "num_selected": len(state.selected_evidence),
            "num_removed": removed_count,
            "num_support_actions": support_count,
            "num_contradict_actions": contradict_count,
            "entropy": getattr(policy, "last_entropy", None),
        })

        tracker.save_trajectory(ep, trajectory)

        print(f"Episode {ep} | Reward: {round(total_reward, 3)}")

    tracker.save_summary()

    print("\nSaved experiment to:", tracker.get_dir())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--method", type=str, default="ppo")
    args = parser.parse_args()

    train(args.episodes, args.method)


if __name__ == "__main__":
    main()