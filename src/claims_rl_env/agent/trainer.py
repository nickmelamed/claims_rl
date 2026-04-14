class Trainer:
    def __init__(self, env, policy):
        self.env = env
        self.policy = policy

    def run_episode(self):
        state = self.env.reset()
        done = False

        while not done:
            action, payload = self.policy.act(state)
            state, reward, done, _ = self.env.step(action, payload)

        return reward

    def train(self, episodes=100):
        rewards = []
        for _ in range(episodes):
            r = self.run_episode()
            rewards.append(r)
        return rewards