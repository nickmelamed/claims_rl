from claims_rl_env.environment.environment import ClaimEnv
from claims_rl_env.environment.curriculum import Curriculum
from claims_rl_env.judge.judge import Judge
from claims_rl_env.agent.policy import RandomPolicy
from claims_rl_env.agent.trainer import Trainer
from claims_rl_env.data.dataset import load_dataset


def main():
    dataset = load_dataset()
    curriculum = Curriculum()
    judge = Judge()

    env = ClaimEnv(dataset, judge, curriculum)
    policy = RandomPolicy()
    trainer = Trainer(env, policy)

    rewards = trainer.train(episodes=50)

    print("Avg reward:", sum(rewards) / len(rewards))


if __name__ == "__main__":
    main()