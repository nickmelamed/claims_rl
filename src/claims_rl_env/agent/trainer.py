import os
from claims_rl_env.utils.experiment import ExperimentTracker


class Trainer:
    def __init__(self, env, policy, episodes=50, exp_name="default"):
        self.env = env
        self.policy = policy
        self.episodes = episodes

        # Initialize tracker
        self.tracker = ExperimentTracker(exp_name)

        # Save config
        self.tracker.save_config({
            "episodes": episodes,
            "policy": policy.__class__.__name__,
            "environment": env.__class__.__name__,
            "dataset_size": len(env.dataset),
            "max_steps": getattr(env.state, "max_steps", None) if hasattr(env, "state") else None,
            "reward_type": "hybrid_deterministic",
            "type": "PPO" if hasattr(policy, "update") else "static"
            
        })

    def train(self):
        results = []

        for ep in range(self.episodes):
            state = self.env.reset()
            done = False
            total_reward = 0.0
            steps = 0

            trajectory = []  # important for PPO later

            while not done:
                action, payload = self.policy.act(state)
                next_state, reward, done, _ = self.env.step(action, payload)

                # store trajectory (future RL use)
                trajectory.append({
                    "state": state,
                    "action": action,
                    "reward": reward
                })

                state = next_state
                total_reward += reward
                steps += 1

            # metrics
            metrics = {
                "episode": ep,
                "reward": float(total_reward),
                "steps": steps,
                "avg_reward_per_step": float(total_reward / max(steps, 1))
            }

            results.append(metrics)

            # log via tracker 
            self.tracker.log(metrics)

            print(
                f"Episode {ep:03d} | "
                f"Reward: {total_reward:.3f} | "
                f"Steps: {steps}"
            )

            # policy update hook 
            if hasattr(self.policy, "update"):
                self.policy.update(trajectory)

        # finalize experiment
        self.tracker.save() 

        return results
