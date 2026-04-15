from claims_rl_env.environment.environment import ClaimEnv
from claims_rl_env.data.dataset import load_dataset
from claims_rl_env.agent.llm_policy import LLMPolicy
from claims_rl_env.agent.llm_client import LLMClient


def main():
    dataset = load_dataset()

    env = ClaimEnv(dataset)

    llm = LLMClient()
    policy = LLMPolicy(llm)

    state = env.reset()
    done = False

    step = 0

    print("\n=== START EPISODE ===")
    print("CLAIM:", state.claim)

    while not done:
        action, payload = policy.act(state)
        state, reward, done, _ = env.step(action, payload)

        step += 1

        print(f"\n--- STEP {step} ---")
        print("Action:", action)
        print("Payload:", payload)
        print("Selected:", [e.id for e in state.selected_evidence])
        print("Debate:", state.debate_history)

    print("\n=== FINAL ===")
    print("Steps:", step)
    print("Final Reward:", round(reward, 3))


if __name__ == "__main__":
    main()