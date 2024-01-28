from PromptInputs import Mitigation
from ExamplesGenerator import ExamplesGenerator
from TestModelAccuracy import TestModelAccuracy
from LLMCaller import OpenAILLM

from example_risk_assessments import example_risk_assessments
from example_mitigations import correct_mitigation_examples_list, MitigationExamplesGenerator
from ExamplesGenerator import RiskAssessmentExamplesGenerator

if __name__ == '__main__':
    # examples_generator = MitigationExamplesGenerator(correct_examples_list=correct_mitigation_examples_list)
    examples_generator = RiskAssessmentExamplesGenerator(risk_assessments=example_risk_assessments,
                                                         ground_truth_parameter='mitigation_prompt_expected_output',
                                                        method_to_get_prompt_input='get_mitigation_input')
    
    examples = examples_generator.get_input_and_expected_output_list()
    
    test_accuracy = TestModelAccuracy(test_description="""Testing mitigation input in student Fluids Lab and TPS Risk Assessment examples.
                                    
                                    Tried using low temperature (0.1) and high number of max_tokens (400)
                                      """,
                                      LLM=OpenAILLM(),
                                                LLM_name='gpt-3.5-turbo',
                                                list_of_input_and_expected_outputs=examples,
                                                sheet_name='Mitigation In Context')
    test_accuracy.run_test()