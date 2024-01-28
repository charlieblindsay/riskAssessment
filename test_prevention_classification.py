from example_preventions import correct_prevention_examples_list
from example_mitigations import correct_mitigation_examples_list

from ExamplesGenerator import ExamplesGenerator, InputAndExpectedOutputGenerator
from TestModelAccuracy import TestModelAccuracy
from PromptInputs import PreventionClassification
from LLMCaller import OpenAILLM

from example_preventions import correct_prevention_examples_list
from example_mitigations import correct_mitigation_examples_list

class GeneratePreventionClassificationExamples(ExamplesGenerator):
    def __init__(self, correct_prevention_examples_list, correct_mitigation_examples_list):
        self.correct_prevention_examples_list = correct_prevention_examples_list
        self.correct_mitigation_examples_list = correct_mitigation_examples_list

        self.correct_examples_list = self.generate_correct_examples_from_prevention_list()
        self.incorrect_examples_list = self.generate_incorrect_examples_from_mitigation_list()

    def generate_correct_example(self, prevention_object):
        return PreventionClassification(
            prevention=prevention_object.prevention,
            activity=prevention_object.activity,
            hazard=prevention_object.hazard,
            how_it_harms=prevention_object.how_it_harms,
            who_it_harms=prevention_object.who_it_harms
        )
    
    def generate_incorrect_example(self, mitigation_object):
        return PreventionClassification(
            prevention=mitigation_object.mitigation,
            activity=mitigation_object.activity,
            hazard=mitigation_object.hazard,
            how_it_harms=mitigation_object.how_it_harms,
            who_it_harms=mitigation_object.who_it_harms
        )
    
    def generate_correct_examples_from_prevention_list(self):
        return [self.generate_correct_example(prevention_object) for prevention_object in self.correct_prevention_examples_list]
    
    def generate_incorrect_examples_from_mitigation_list(self):
        return [self.generate_incorrect_example(mitigation_object) for mitigation_object in self.correct_mitigation_examples_list]

if __name__ == '__main__':
    prevention_classification_examples_generator = GeneratePreventionClassificationExamples(
        correct_prevention_examples_list=correct_prevention_examples_list,
        correct_mitigation_examples_list=correct_mitigation_examples_list)

    examples = prevention_classification_examples_generator.get_input_and_expected_output_list()

    test_accuracy = TestModelAccuracy(
        LLM=OpenAILLM(),
        LLM_name='gpt-3.5-turbo',
        list_of_input_and_expected_outputs=examples,
        sheet_name='Prevention Classification',
        test_description='Evaluating prompt on Chat GPT generated data for activities')
    test_accuracy.run_test()
