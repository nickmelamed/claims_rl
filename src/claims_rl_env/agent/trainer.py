import numpy as np

from claims_rl_env.utils.experiment import ExperimentTracker
from claims_rl_env.agent.bandit import EpsilonGreedyBandit
from claims_rl_env.agent.policy_gradient import PolicyGradient
from claims_rl_env.agent.ppo import PPO
from claims_rl_env.environment.actions import Actions, ACTIONS



class Trainer:
    def __init__(self, env, policy, episodes=50, algo="pg", exp_name="default"):
        self.env = env
        self.policy = policy
        self.episodes = episodes
        self.algo = algo

        # Initialize tracker
        self.tracker = ExperimentTracker(exp_name)

        # initialize RL algos 
        n_actions = len(ACTIONS)

        if algo == 'bandit':
            self.rl = EpsilonGreedyBandit(n_actions=n_actions)

        elif algo == 'pg':
            self.rl = PolicyGradient(policy)

        elif algo == 'ppo':
            self.rl = PPO(policy)

        else:
            raise ValueError(f"Unknown algorithm: {algo}")

        # Save config
        self.tracker.save_config({
            "episodes": episodes,
            "algo": algo,
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

                # action selection 
                if self.algo == 'bandit':
                    action_idx = self.rl.select_action()
                    action = list(Actions)[action_idx]

                    # payload handling
                    if action == Actions.SELECT:
                        doc = np.random.choice(state.evidence_pool)
                        payload = doc.id

                    elif action in [Actions.SUPPORT, Actions.CONTRADICT]:
                        payload = 'Generated argument'
                    else:
                        payload = None
                
                else:
                    action, payload = self.policy.act(state)
                    action_idx = self.policy.actions.index(action)

                # env step
                next_state, reward, done, _ = self.env.step(action, payload)

                # store trajectory
                if self.algo == 'ppo':
                    old_prob = self.policy.get_prob(action_idx)
                    trajectory.append((state, action_idx, old_prob, reward))
                else:
                    trajectory.append((state, action_idx, reward))

                # bandit update 
                if self.algo == 'bandit':
                    self.rl.update(action_idx, reward)

                state = next_state
                total_reward += reward
                steps += 1

            # policy update
            if self.algo == 'pg':
                self.rl.update(trajectory)
            elif self.algo == 'ppo':
                self.rl.update(trajectory)

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


        # finalize experiment
        self.tracker.save() 

        return results
