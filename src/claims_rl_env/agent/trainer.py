import numpy as np
import json

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

        self.tracker = ExperimentTracker(exp_name)

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
            "rl_method": algo,
            "num_episodes": episodes,
            "policy": policy.__class__.__name__,
            "environment": env.__class__.__name__,
            "dataset_size": len(env.dataset),
        })

    def train(self):
        results = []

        for ep in range(self.episodes):
            state = self.env.reset()
            done = False

            total_reward = 0.0
            steps = 0

            # trajectory storage
            rl_trajectory = []
            viz_trajectory = []

            # behavior tracking
            support_count = 0
            contradict_count = 0
            removed_count = 0

            while not done:

                # action selection 
                if self.algo == 'bandit':
                    action_idx = self.rl.select_action()
                    action = list(Actions)[action_idx]

                    if action == Actions.SELECT:
                        doc = np.random.choice(state.evidence_pool)
                        payload = doc.id
                    elif action in [Actions.SUPPORT, Actions.CONTRADICT]:
                        payload = "Generated argument"
                    else:
                        payload = None

                else:
                    action, payload = self.policy.act(state)
                    action_idx = self.policy.actions.index(action)

                # env step
                next_state, reward, done, _ = self.env.step(action, payload)

                # track behavior 
                if action == Actions.SUPPORT or action == "generate_support_argument":
                    support_count += 1
                elif action == Actions.CONTRADICT or action == "generate_contradict_argument":
                    contradict_count += 1
                elif action == Actions.REMOVE or action == "remove_evidence":
                    removed_count += 1

                probs = getattr(self.policy, "last_probs", None)

                # trajectory entry
                argument = None
                evidence_ids = None

                if isinstance(payload, dict):
                    argument = payload.get("argument")
                    evidence_ids = payload.get("evidence")

                viz_trajectory.append({
                    "step": steps + 1,
                    "action": str(action),
                    "reward": float(reward),
                    "argument": argument,
                    "evidence_used": evidence_ids,
                    "selected_ids": [e.id for e in next_state.selected_evidence],
                    "action_probs": probs.tolist() if probs is not None else None,
                    "action_names": [str(a) for a in ACTIONS],
                    "entropy": getattr(self.policy, "last_entropy", None),

                    # evidence + claim 
                    "claim": state.claim,
                    "evidence_pool": [
                        {"id": e.id, "text": e.text}
                        for e in state.evidence_pool
                    ]
                })

                # RL storage 
                if self.algo == 'ppo':
                    old_prob = self.policy.get_probs()[action_idx]  # 🔥 FIX
                    rl_trajectory.append((state, action_idx, old_prob, reward))

                elif self.algo == 'pg':
                    rl_trajectory.append((state, action_idx, reward))

                elif self.algo == 'bandit':
                    self.rl.update(action_idx, reward)

                state = next_state
                total_reward += reward
                steps += 1

            # policy update 
            if self.algo == 'pg':
                self.rl.update(rl_trajectory)

            elif self.algo == 'ppo':
                self.rl.update(rl_trajectory)

            # metrics logging 
            metrics = {
                "episode": ep,
                "reward": float(total_reward),
                "num_steps": steps,
                "final_decision": getattr(state, "final_decision", None),
                "correct": getattr(state, "correct", None),
                "num_selected": len(state.selected_evidence),
                "num_removed": removed_count,
                "num_support_actions": support_count,
                "num_contradict_actions": contradict_count,
                "entropy": getattr(self.policy, "last_entropy", None),
            }

            results.append(metrics)

            self.tracker.log_episode(metrics)

            # save trajectory
            self.tracker.save_trajectory(ep, viz_trajectory)

            print(
                f"Episode {ep:03d} | "
                f"Reward: {total_reward:.3f} | "
                f"Steps: {steps}"
            )

        self.tracker.save_summary()

        return results