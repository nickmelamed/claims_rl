from claims_rl_env.utils.experiment import ExperimentTracker

class Trainer:
    def __init__(self, environment, policy, judge, episodes=100, exp_name="default"):
        self.env = environment
        self.policy = policy
        self.judge = judge
        self.episodes = episodes

        self.tracker = ExperimentTracker(exp_name)

    def train(self):
        results = []

        for episode in range(self.episodes):
            state = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                action = self.policy.act(state)
                state, reward, done, _ = self.env.step(action)
                total_reward += reward

            metrics = {
                "episode": episode,
                "reward": total_reward,
                #"steps": step_count,
                "success": done,
            }

            self.tracker.log(metrics)
            results.append(metrics)

        return results