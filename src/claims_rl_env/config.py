from claims_rl_env.environment.environment import Environment
from claims_rl_env.agent.policy import Policy
from claims_rl_env.agent.trainer import Trainer
from claims_rl_env.judge.judge import Judge


def build_environment(config: dict = None):
    config = config or {}

    return Environment(
        max_steps=config.get("max_steps", 10),
        curriculum=config.get("curriculum", None),
    )


def build_policy(config: dict = None):
    config = config or {}

    return Policy(
        temperature=config.get("temperature", 0.7),
    )


def build_judge(config: dict = None):
    config = config or {}

    return Judge(
        reward_scale=config.get("reward_scale", 1.0),
    )


def build_trainer(config: dict = None):
    config = config or {}

    env = build_environment(config)
    policy = build_policy(config)
    judge = build_judge(config)

    return Trainer(
        environment=env,
        policy=policy,
        judge=judge,
        episodes=config.get("episodes", 100),
    )