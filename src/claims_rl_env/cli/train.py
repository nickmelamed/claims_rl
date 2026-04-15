from claims_rl_env.environment.environment import ClaimEnv
from claims_rl_env.environment.curriculum import Curriculum
from claims_rl_env.agent.policy import SoftmaxPolicy
from claims_rl_env.agent.trainer import Trainer
from claims_rl_env.environment.actions import ACTIONS
from claims_rl_env.data.dataset import load_dataset

import argparse

def train(episodes):
    dataset = load_dataset()
    curriculum = Curriculum()

    # Apply curriculum sampling inside env
    sampled_dataset = [curriculum.sample(dataset) for _ in range(len(dataset))]

    env = ClaimEnv(sampled_dataset)

    n_actions = len(list(ACTIONS))
    policy = SoftmaxPolicy(n_actions)

    trainer = Trainer(env, policy, episodes=episodes)

    results = trainer.train()

    avg_reward = sum(r["reward"] for r in results) / len(results)

    print("\n=== TRAINING COMPLETE ===")
    print("Avg reward:", round(avg_reward, 3))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args()

    train(args.episodes)

if __name__ == "__main__":
    main()