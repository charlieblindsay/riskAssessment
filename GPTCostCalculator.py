# NOTE: On huggingface, "The Inference API is free to use, and rate limited."
# However, to use advanced LLMs like Llama, it costs $9 (or £7.14) per month.
# My subscription started on the 13th of November 2023

import nltk

class GPTCostCalculator:
    def __init__(self):
        self.cost_per_input_token = 0.001 / 1000
        self.cost_per_output_token = 0.002 / 1000
        nltk.download('punkt')

    def calculate_number_of_tokens(self, prompt):

        # Tokenize the input string into words
        tokens = nltk.word_tokenize(prompt)
        
        # Count the number of tokens
        num_tokens = len(tokens)
        
        return num_tokens

    def calculate_cost(self, prompt_input, prompt_output):
        num_input_tokens = self.calculate_number_of_tokens(prompt_input)
        num_output_tokens = self.calculate_number_of_tokens(prompt_output)

        return self.cost_per_input_token * num_input_tokens + self.cost_per_output_token * num_output_tokens