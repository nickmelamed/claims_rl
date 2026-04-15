import json
from claims_rl_env.environment.actions import Actions

class LLMPolicy:
    def __init__(self, llm):
        self.llm = llm

    def build_prompt(self, state):
        evidence = "\n".join([f"[{e.id}] {e.text}" for e in state.evidence_pool])
        selected = [e.id for e in state.selected_evidence]

        return f"""
CLAIM: {state.claim}

EVIDENCE:
{evidence}

SELECTED: {selected}
DEBATE: {state.debate_history}

Choose ONE action:
select_evidence(id)
remove_evidence(id)
generate_support_argument(text)
generate_contradict_argument(text)
finalize

Return JSON:
{{"action": "...", "payload": ...}}
"""

    def parse(self, text):
        try:
            data = json.loads(text)
            return data["action"], data.get("payload")
        except:
            return Actions.FINALIZE, None

    def act(self, state):
        prompt = self.build_prompt(state)
        response = self.llm.generate(prompt)
        return self.parse(response)
