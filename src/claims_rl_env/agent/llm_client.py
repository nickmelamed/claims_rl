from transformers import pipeline

class LLMClient:
    def __init__(self, model_name="microsoft/phi-2"):
        self.pipe = pipeline("text-generation", model=model_name)

    def generate(self, prompt):
        out = self.pipe(prompt, max_new_tokens=128)[0]["generated_text"]
        return out