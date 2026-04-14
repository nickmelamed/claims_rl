import json
from claims_rl_env.environment.actions import Actions


class LLMPolicy:
    def __init__(self, llm_client):
        self.llm = llm_client

    def build_prompt(self, state):
        evidence_list = "\n".join([
            f"[{e.id}] ({e.label}) {e.text}"
            for e in state.evidence_pool
        ])

        selected = [e.id for e in state.selected_evidence]

        prompt = f"""
You are an RL agent solving a claim verification task.

CLAIM:
{state.claim}

AVAILABLE EVIDENCE:
{evidence_list}

SELECTED EVIDENCE IDS:
{selected}

DEBATE HISTORY:
{state.debate_history}

You may take ONE action:

- select_evidence(id)
- remove_evidence(id)
- generate_support_argument(text)
- generate_contradict_argument(text)
- finalize

Respond ONLY in JSON:
{{
  "action": "...",
  "payload": ...
}}
"""
        return prompt

    def parse_action(self, response_text):
        try:
            data = json.loads(response_text)
            return data["action"], data.get("payload", None)
        except:
            return Actions.FINALIZE, None

    def act(self, state):
        prompt = self.build_prompt(state)
        response = self.llm.generate(prompt)

        return self.parse_action(response)