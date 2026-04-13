from env.environment import ClaimEnv
from env.curriculum import Curriculum
from judge.judge import Judge
from agent.policy import RandomPolicy
from agent.trainer import Trainer
from data.dataset import load_dataset


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