class LLMJudge:
    def evaluate_reasoning(self, reasoning):
        # placeholder (replace with real LLM call)
        return {
            "LCS": min(1.0, len(reasoning) / 200),
            "HLS": 0.2
        }