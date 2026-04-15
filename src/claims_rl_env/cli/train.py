from claims_rl_env.environment.environment import ClaimEnv
from claims_rl_env.environment.curriculum import Curriculum
from claims_rl_env.agent.policy import RandomPolicy
from claims_rl_env.agent.trainer import Trainer
from claims_rl_env.data.dataset import load_dataset


def main():
    dataset = load_dataset()
    curriculum = Curriculum()

    # Apply curriculum sampling inside env
    sampled_dataset = [curriculum.sample(dataset) for _ in range(len(dataset))]

    env = ClaimEnv(sampled_dataset)
    policy = RandomPolicy()

    trainer = Trainer(env, policy, episodes=50)

    results = trainer.train()

    avg_reward = sum(r["reward"] for r in results) / len(results)

    print("\n=== TRAINING COMPLETE ===")
    print("Avg reward:", round(avg_reward, 3))


if __name__ == "__main__":
    main()