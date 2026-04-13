class DummyLLM:
    def generate(self, prompt):
        print("\n===== PROMPT =====\n")
        print(prompt)

        # manual input for testing
        response = input("\nEnter JSON action: ")
        return response