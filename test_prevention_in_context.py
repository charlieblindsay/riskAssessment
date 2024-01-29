from PromptInputs import Prevention
from ExamplesGenerator import ExamplesGenerator
from TestModelAccuracy import TestModelAccuracy
from LLMCaller import OpenAILLM
from example_risk_assessments import example_risk_assessments

from example_preventions import correct_prevention_examples_list, PreventionExamplesGenerator
from ExamplesGenerator import RiskAssessmentExamplesGenerator

if __name__ == '__main__':
    # examples_generator = PreventionExamplesGenerator(correct_examples_list=correct_prevention_examples_list)
    examples_generator = RiskAssessmentExamplesGenerator(risk_assessments=example_risk_assessments,
                                                         ground_truth_parameter='prevention_prompt_expected_output',
                                                        method_to_get_prompt_input='get_prevention_input')
    
    examples = examples_generator.get_input_and_expected_output_list()

    test_accuracy = TestModelAccuracy(test_description="""Testing prevention input in student Fluids Lab and TPS presentation Risk Assessment examples.
                                      
                                      Added hazard event description.

                                      Refer to hazard event in mitigation and prevention instructions.

                                      Added 6 new few shot examples, 3 of which are prevention and 3 of which are mitigation. 
                                       """,
                                      LLM=OpenAILLM(),
                                                LLM_name='gpt-3.5-turbo',
                                                list_of_input_and_expected_outputs=examples,
                                                sheet_name='Prevention In Context')
    test_accuracy.run_test()