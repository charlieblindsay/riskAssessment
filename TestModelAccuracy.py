import re
import csv
import pathlib
from datetime import datetime

from LLMCaller import LLMCaller, LLMWithCandidateLabels, LLMWithGeneratedText, OpenAILLM
from GPTCostCalculator import GPTCostCalculator
from InputAndExpectedOutput import InputAndExpectedOutput
from GoogleSheetsWriter import GoogleSheetsWriter
from RegexPatternMatcher import RegexPatternMatcher

class TestModelAccuracy:
    def __init__(self, 
                LLM: LLMCaller,
                LLM_name: str,
                list_of_input_and_expected_outputs: list[InputAndExpectedOutput],
                sheet_name: str,
                test_description: str):
        
        self.LLM = LLM
        self.LLM_name = LLM_name
        self.list_of_input_and_expected_outputs = list_of_input_and_expected_outputs
        self.sheet_name = sheet_name
        self.test_description = test_description
        self.spreadsheet_id = '1d7Tq7qEaNTrhm1E7qcGvl3Dkr8cFNdSpOul9RezjVs4'

    def get_prompt_input_and_output(self, input_and_expected_output):
        input = input_and_expected_output.input

        prompt_input = self.LLM.get_prompt_input(prompt_input=input)
        prompt_output = self.LLM.get_model_output(prompt_input=input)
        
        return prompt_input, prompt_output

    def convert_list_of_lists_to_string(self, list_of_lists):
        # Convert each sublist to a string with newline
        sublist_strings = ['[' + ', '.join(map(str, sublist)) + ']' for sublist in list_of_lists]

        # Join the sublist strings with a newline and add outer brackets
        return '[' + ',\n'.join(sublist_strings) + ']'

    def get_model_accuracy_and_model_outputs(self):
        print('Counting correct responses...')
        count_correct = 0
        count_true_positives = 0
        count_false_positives = 0
        count_false_negatives = 0
        count_true_negatives = 0

        output_string = ''
        prompt_outputs_for_correct_responses = ''
        prompt_outputs_for_incorrect_responses = ''

        for i in range(len(self.list_of_input_and_expected_outputs)):
            input = self.list_of_input_and_expected_outputs[i].input

            prompt_input, prompt_output = self.get_prompt_input_and_output(self.list_of_input_and_expected_outputs[i])
            regex_pattern_matcher = RegexPatternMatcher()

            pattern_matching_method_string = input.pattern_matching_method
            pattern_matching_method = getattr(regex_pattern_matcher, input.pattern_matching_method)

            if hasattr(regex_pattern_matcher, pattern_matching_method_string) and callable(pattern_matching_method):

                pattern_matched = pattern_matching_method(prompt_output)
                expected_output = self.list_of_input_and_expected_outputs[i].expected_output

                result_dict = {'input': self.list_of_input_and_expected_outputs[i].input.to_string(),
                                'pattern_matched': pattern_matched, 
                                'expected_output': expected_output}
                
                output_string += f'{i + 1}: {str(result_dict)}\n\n'

                if pattern_matched == expected_output:
                    count_correct += 1

                    if pattern_matching_method_string == "check_string_for_true_or_false":
                        if pattern_matched == True:
                            count_true_positives += 1
                        else:
                            count_true_negatives += 1

                    prompt_outputs_for_correct_responses += f'{i + 1}: {prompt_output}\nExpected output: {expected_output}\n\n'
                
                else:

                    if pattern_matching_method_string == "check_string_for_true_or_false":
                        if pattern_matched == True:
                            count_false_positives += 1
                        else:  
                            count_false_negatives += 1

                    prompt_outputs_for_incorrect_responses += f'{i + 1}: {prompt_output}\nExpected output: {expected_output}\n\n'
                
                print(count_correct)
        accuracy = round(count_correct / len(self.list_of_input_and_expected_outputs) * 100, 2)

        if pattern_matching_method_string == "check_string_for_true_or_false":
            confusion_matrix = self.convert_list_of_lists_to_string([[count_true_positives, count_false_negatives], [count_false_positives, count_true_negatives]])
        else:
            confusion_matrix = f'Confusion matrix not yet created for multiclass classification\nNumber correct = {count_correct}'

        return accuracy, confusion_matrix, output_string, prompt_outputs_for_correct_responses, prompt_outputs_for_incorrect_responses
    
    def get_first_prompt_input_and_output(self):
        first_input = self.list_of_input_and_expected_outputs[0].input
        first_prompt_input = self.LLM.get_prompt_input(prompt_input=first_input)
        first_prompt_output = self.LLM.get_model_output(prompt_input=first_input)

        return first_prompt_input, first_prompt_output
    
    def total_cost_of_calling_LLM_API(self):
        GPT_cost_calculator = GPTCostCalculator()
        
        if self.LLM_name == 'gpt-3.5-turbo':
            total_cost_of_calling_LLM_API = 0

            for input_and_expected_output in self.list_of_input_and_expected_outputs:
                prompt_input, prompt_output = self.get_prompt_input_and_output(input_and_expected_output)
                total_cost_of_calling_LLM_API = total_cost_of_calling_LLM_API + GPT_cost_calculator.calculate_cost(prompt_input=prompt_input, prompt_output=prompt_output)
            
            return total_cost_of_calling_LLM_API
        else:
            return 0
    
    def save_test_results(self, 
                          accuracy, 
                          confusion_matrix,
                          output_string, 
                          prompt_outputs_for_correct_responses,
                          prompt_outputs_for_incorrect_responses,
                          first_prompt_input,
                          first_prompt_output):

        datetime_now = datetime.now().strftime("%d-%m_%H-%M")

        if self.LLM_name == 'gpt-3.5-turbo':
            model_parameters = f'Temp: {self.LLM.temperature}, Max tokens: {self.LLM.max_tokens}'
        else:
            model_parameters = ''
        
        num_examples = len(self.list_of_input_and_expected_outputs)

        # total_cost_of_calling_LLM_API = self.total_cost_of_calling_LLM_API()

        new_line_data = [self.test_description, 
                         self.LLM_name, 
                            datetime_now, 
                            0,
                            model_parameters, 
                            accuracy, 
                            num_examples,
                            confusion_matrix,
                            output_string,
                            first_prompt_input,
                            first_prompt_output,
                            prompt_outputs_for_correct_responses,
                            prompt_outputs_for_incorrect_responses]
        
        sheets_writer = GoogleSheetsWriter(sheet_name=self.sheet_name, spreadsheet_id=self.spreadsheet_id)
        sheets_writer.write_to_sheets(new_line_data)
        
    # TODO: As below, remove positional arguments and only use keyword arguments
    def run_test(self):
        accuracy, confusion_matrix, output_string, prompt_outputs_for_correct_responses, prompt_outputs_for_incorrect_responses = self.get_model_accuracy_and_model_outputs()
        first_prompt_input, first_prompt_output = self.get_first_prompt_input_and_output()
        self.save_test_results(accuracy=accuracy, 
                               confusion_matrix=confusion_matrix,
                               output_string=output_string[:40000], 
                               first_prompt_input=first_prompt_input,
                               first_prompt_output=first_prompt_output,
                               prompt_outputs_for_correct_responses=prompt_outputs_for_correct_responses[:40000],
                               prompt_outputs_for_incorrect_responses=prompt_outputs_for_incorrect_responses[:40000],)

class TestModelAccuracyForActivitiesWithLLAMA(TestModelAccuracy):
    def get_regex_pattern(self, input_and_expected_output):
        input = input_and_expected_output.input
        activity = input.activity
        return re.compile(r"Output: dict(input\s*=\s*\"{}\", is_activity\s*=\s*(true|false))".format(re.escape(activity)), re.IGNORECASE)