from claims_rl_env.environment.environment import ClaimEnv
from claims_rl_env.environment.curriculum import Curriculum
from claims_rl_env.agent.policy import SoftmaxPolicy, ActorCriticPolicy
from claims_rl_env.environment.actions import ACTIONS
from claims_rl_env.data.dataset import load_dataset
from claims_rl_env.agent.trainer import Trainer
from claims_rl_env.agent.config import PPOConfig, PGConfig, BanditConfig

import argparse


def train(episodes, method="ppo", policy='actor'):
    dataset = load_dataset()
    curriculum = Curriculum()

    sampled_dataset = [curriculum.sample(dataset) for _ in range(len(dataset))]
    env = ClaimEnv(sampled_dataset)

    if policy == 'actor':
        policy = ActorCriticPolicy(len(list(ACTIONS)))
    elif policy == 'softmax':
        policy = SoftmaxPolicy(len(list(ACTIONS)))

    if method == 'ppo':
        config = PPOConfig()
    elif method == 'pg':
        config = PGConfig()
    elif method == 'bandit':
        config = BanditConfig()

    trainer = Trainer(
        env=env,
        policy=policy,
        config=config,
        episodes=episodes,
        algo=method,
        exp_name=f"{method}_run"
    )

    trainer.train()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--method", type=str, default="ppo")
    parser.add_argument("--policy", type=str, default='actor')
    args = parser.parse_args()

    train(args.episodes, args.method)


if __name__ == "__main__":
    main()