from env.environment import ClaimEnv
from env.curriculum import Curriculum
from judge.judge import Judge
from data.dataset import load_dataset
from agent.llm_policy import LLMPolicy
from agent.llm_client import DummyLLM


def main():
    dataset = load_dataset()
    curriculum = Curriculum()
    judge = Judge()

    env = ClaimEnv(dataset, judge, curriculum)

    llm = DummyLLM()
    policy = LLMPolicy(llm)

    state = env.reset()
    done = False

    while not done:
        action, payload = policy.act(state)
        state, reward, done, _ = env.step(action, payload)

        print("\n--- STEP RESULT ---")
        print("Action:", action)
        print("Selected:", [e.id for e in state.selected_evidence])
        print("Debate:", state.debate_history)

    print("\n=== FINAL REWARD ===", reward)


if __name__ == "__main__":
    main()