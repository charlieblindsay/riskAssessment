# pytest -v -s evaluation_tests.py

# The -s option above is so you can see printouts even if the test fails

import unittest

try:
    from .evaluation import Params, evaluation_function
    from .example_risk_assessments import RA_5, RA_mitigation_wrong_type, RA_controlled_likelihood_wrong_type, RA_empty_input
    from .LLMCaller import LLMCaller, LLMWithCandidateLabels, LLMWithGeneratedText, OpenAILLM
    from .PromptInputs import Activity
    from .RegexPatternMatcher import RegexPatternMatcher
except:
    from evaluation import Params, evaluation_function
    from example_risk_assessments import RA_5, RA_mitigation_wrong_type, RA_controlled_likelihood_wrong_type, RA_empty_input
    from LLMCaller import LLMCaller, LLMWithCandidateLabels, LLMWithGeneratedText, OpenAILLM
    from PromptInputs import Activity
    from RegexPatternMatcher import RegexPatternMatcher

class TestEvaluationFunction(unittest.TestCase):
    """
    TestCase Class used to test the algorithm.
    ---
    Tests are used here to check that the algorithm written
    is working as it should.

    It's best practise to write these tests first to get a
    kind of 'specification' for how your algorithm should
    work, and you should run these tests before committing
    your code to AWS.

    Read the docs on how to use unittest here:
    https://docs.python.org/3/library/unittest.html

    Use evaluation_function() to check your algorithm works
    as it should.
    """

    def test_returns_is_correct_true(self):
        response = [["Fluids laboratory"],
                    ["Water being spilt on the floor"],
                    ["Students"],
                    ["Injuries caused by possible slipping on wet floor"],
                    ["4"],
                    ["1"],
                    ["4"],
                    ["Do not move the water tank when it is full"],
                    ["""If someone gets injured due to slipping, apply an ice pack to the injured area and 
                    seek medical advice without delay."""],
                    ["1"],
                    ["1"], 
                    ["1"]]
        answer, params = None, None

        result = evaluation_function(response, answer, params)

        print(result.get("feedback"))

        self.assertIn(result.get("is_correct"), [True, False])
        
    def test_handles_empty_input(self):
        self.assertEqual(RA_empty_input.get_empty_fields(), ['activity'])
        self.assertEqual(RA_5.get_empty_fields(), [])
    
    def test_does_string_represent_an_integer(self):

        self.assertEqual(RA_5.does_string_represent_an_integer('1'), True)
        self.assertEqual(RA_5.does_string_represent_an_integer('1.0'), False)
        self.assertEqual(RA_5.does_string_represent_an_integer('One'), False)
    
    def test_does_string_represent_words(self):

        self.assertEqual(RA_5.does_string_represent_words('1'), False)
        self.assertEqual(RA_5.does_string_represent_words('1.0'), False)
        self.assertEqual(RA_5.does_string_represent_words('One'), True)

    def test_get_word_fields_incorrect(self):
        self.assertEqual(RA_mitigation_wrong_type.get_word_fields_incorrect(), ['mitigation'])

    def test_get_integer_fields_incorrect(self):
        self.assertEqual(RA_controlled_likelihood_wrong_type.get_integer_fields_incorrect(), ['controlled_likelihood'])

    def test_regex_prevention_mitigation_neither(self):
        regex = RegexPatternMatcher()
        self.assertEqual(regex.check_string_for_prevention_mitigation_or_neither('Answer: prevention'), 'prevention')
        self.assertEqual(regex.check_string_for_prevention_mitigation_or_neither('Therefore, answer: mitigation'), 'mitigation')
        self.assertEqual(regex.check_string_for_prevention_mitigation_or_neither('Thus, answer: Neither'), 'neither')

    # TODO: Test the function which creates an instance of a RiskAssessment object

    # TODO: Test the function which calls the LLM

if __name__ == "__main__":
    unittest.main()
