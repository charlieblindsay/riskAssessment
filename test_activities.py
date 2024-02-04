# pytest -v -s local_tests.py

# The -s option above is so you can see printouts even if the test fails

from TestModelAccuracy import TestModelAccuracy, TestModelAccuracyForActivitiesWithLLAMA
from LLMCaller import OpenAILLM, LLMWithGeneratedText

import re

try:
    from InputAndExpectedOutput import InputAndExpectedOutput
    from LLMCaller import *
    from PromptInputs import Activity
except ImportError:
    from .InputAndExpectedOutput import InputAndExpectedOutput
    from .LLMCaller import *
    from .PromptInputs import Activity

activities = [
    InputAndExpectedOutput(input=Activity(activity='dog'), expected_output=False),
    InputAndExpectedOutput(input=Activity(activity='Butterfly spotting'), expected_output=True),
    InputAndExpectedOutput(input=Activity(activity='Climbing a mountain'), expected_output=True),
    InputAndExpectedOutput(input=Activity(activity='Paris'), expected_output=False),
    InputAndExpectedOutput(input=Activity(activity='table'), expected_output=False),
    InputAndExpectedOutput(input=Activity(activity='cat'), expected_output=False),
    InputAndExpectedOutput(input=Activity(activity='water'), expected_output=False),
    InputAndExpectedOutput(input=Activity(activity='team'), expected_output=False),
    InputAndExpectedOutput(input=Activity(activity='toothpaste'), expected_output=False),
    InputAndExpectedOutput(input=Activity(activity='John\'s car'), expected_output=False),
    InputAndExpectedOutput(input=Activity(activity='he'), expected_output=False),
    InputAndExpectedOutput(input=Activity(activity='swimming'), expected_output=True),
    InputAndExpectedOutput(input=Activity(activity='Cooking a meal'), expected_output=True),
    InputAndExpectedOutput(input=Activity(activity='Painting a picture'), expected_output=True),
    InputAndExpectedOutput(input=Activity(activity='Swimming in a pool'), expected_output=True),
    InputAndExpectedOutput(input=Activity(activity='Writing a letter'), expected_output=True),
    InputAndExpectedOutput(input=Activity(activity='Riding a bike'), expected_output=True),
    InputAndExpectedOutput(input=Activity(activity='Singing a song'), expected_output=True),
    InputAndExpectedOutput(input=Activity(activity='Building a sandcastle'), expected_output=True),
    InputAndExpectedOutput(input=Activity(activity='Planting flowers'), expected_output=True),
    InputAndExpectedOutput(input=Activity(activity='Chess'), expected_output=True),
    InputAndExpectedOutput(input=Activity(activity='Marathon'), expected_output=True),
    InputAndExpectedOutput(input=Activity(activity='Puzzle'), expected_output=True),
    InputAndExpectedOutput(input=Activity(activity='Party'), expected_output=True),
    InputAndExpectedOutput(input=Activity(activity='Movie'), expected_output=True)
]

if __name__ == "__main__":
    
    test_accuracy = TestModelAccuracy(
        LLM=OpenAILLM(),
        LLM_name='gpt-3.5-turbo',
        list_of_input_and_expected_outputs=activities,
        sheet_name='Activities',
        test_description='Evaluating prompt on Chat GPT generated data for activities'
        )
    test_accuracy.run_test()

    # Llama_model_name = 'Llama-2-7b-hf'
    # test_accuracy = TestModelAccuracyForActivitiesWithLLAMA(
    #     LLM=LLMWithGeneratedText(
    #         LLM_API_ENDPOINT=f"https://api-inference.huggingface.co/models/meta-llama/{Llama_model_name}"
    #         ),
    #     LLM_name=Llama_model_name,
    #     list_of_input_and_expected_outputs=activities,
    #     folder_name='activities')
    # test_accuracy.run_test()