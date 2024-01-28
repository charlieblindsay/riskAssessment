from typing import Type
from PromptInputs import PromptInput

class CombineAndFlattenListsOfPromptInputs:
    def __init__(self, prompt_input_class: Type[PromptInput]):
        self.prompt_input_class = prompt_input_class
        
    def create_prompt_input_objects(self, *lists):
        combined_list = []

        for sublist in lists:
            combined_list.extend(sublist)

        prompt_input_objects = []

        for item in combined_list:
            prompt_input_objects.append(self.prompt_input_class(item))

        return prompt_input_objects