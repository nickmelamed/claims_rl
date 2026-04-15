from claims_rl_env.judge.metrics import (
    compute_ess,
    compute_ecs,
    compute_lcs,
    compute_hls,
    compute_adversarial_penalty
)


class RewardFunction:
    def compute(self, state, final_output):
        """
        Final_output must contain:
        - reasoning (str)
        - confidence (float)
        - true_score (float)
        """

        # core metrics
        ess = compute_ess(state.selected_evidence)
        ecs = compute_ecs(state.selected_evidence)
        lcs = compute_lcs(state.debate_history)

        # pass reasoning text, not dict
        reasoning = final_output.get("reasoning", "")
        hls = compute_hls(reasoning)

        adversarial_penalty = compute_adversarial_penalty(state.selected_evidence)

        # uncertainty calibration
        confidence = final_output.get("confidence", 0.5)
        true_score = final_output.get("true_score", 0.5)

        uncertainty_penalty = abs(confidence - true_score)

        # reward aggregation
        reward = (
            0.35 * ess
            + 0.25 * (1 - ecs)
            + 0.15 * lcs
            - 0.10 * hls
            - 0.15 * adversarial_penalty
            - 0.20 * uncertainty_penalty
        )

        # anti-gaming penalties
        if len(state.selected_evidence) > 5:
            reward -= 0.2

        # Penalize no reasoning
        if len(reasoning.strip()) == 0:
            reward -= 0.1

        return max(0.0, min(1.0, reward))