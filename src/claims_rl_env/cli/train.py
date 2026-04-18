from claims_rl_env.environment.environment import ClaimEnv
from claims_rl_env.environment.curriculum import Curriculum
from claims_rl_env.agent.policy import SoftmaxPolicy
from claims_rl_env.environment.actions import ACTIONS
from claims_rl_env.data.dataset import load_dataset
from claims_rl_env.agent.trainer import Trainer

import argparse


def train(episodes, method="ppo"):
    dataset = load_dataset()
    curriculum = Curriculum()

    sampled_dataset = [curriculum.sample(dataset) for _ in range(len(dataset))]
    env = ClaimEnv(sampled_dataset)

    policy = SoftmaxPolicy(len(list(ACTIONS)))

    trainer = Trainer(
        env=env,
        policy=policy,
        episodes=episodes,
        algo=method,
        exp_name=f"{method}_run"
    )

    trainer.train()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--method", type=str, default="ppo")
    args = parser.parse_args()

    train(args.episodes, args.method)


if __name__ == "__main__":
    main()