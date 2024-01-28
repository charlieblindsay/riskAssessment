from PromptInputs import Prevention
from ExamplesGenerator import ExamplesGeneratorFromCorrectExamples

class PreventionExamplesGenerator(ExamplesGeneratorFromCorrectExamples):
    def generate_incorrect_example(self, correct_index, incorrect_index):
        return Prevention(
                prevention=self.correct_examples_list[incorrect_index].prevention,
                activity=self.correct_examples_list[correct_index].activity, 
                hazard=self.correct_examples_list[correct_index].hazard, 
                how_it_harms=self.correct_examples_list[correct_index].how_it_harms,
                who_it_harms=self.correct_examples_list[correct_index].who_it_harms)

correct_prevention_examples_list = [
    Prevention(
        prevention="Strict safety protocols for handling and transporting chemicals",
        activity="Industrial chemical production and storage",
        hazard="Industrial Chemical Spill",
        how_it_harms="Environmental contamination",
        who_it_harms="Surrounding ecosystems"
    ),
    # Prevention(
    #     prevention="Zoning regulations restricting construction in flood-prone areas",
    #     activity="Urbanization in flood-prone areas",
    #     hazard="Flood",
    #     how_it_harms="Property damage",
    #     who_it_harms="Residents in affected areas"
    # ),
    # Prevention(
    #     prevention="Strict international health protocols for travel",
    #     activity="Global travel without disease prevention measures",
    #     hazard="Epidemic/Pandemic",
    #     how_it_harms="Spread of infectious diseases",
    #     who_it_harms="Travelers"
    # ),
    # Prevention(
    #     prevention="Sustainable land-use practices",
    #     activity="Deforestation for agriculture",
    #     hazard="Wildfires",
    #     how_it_harms="Habitat destruction",
    #     who_it_harms="Wildlife"
    # ),
    # Prevention(
    #     prevention="Promoting electric vehicles",
    #     activity="Heavy reliance on fossil fuels for transportation",
    #     hazard="Urban Air Pollution",
    #     how_it_harms="Respiratory issues",
    #     who_it_harms="Residents in urban areas"
    # ),
    # Prevention(
    #     prevention="Implementing comprehensive waste disposal",
    #     activity="Industrialization without proper waste management",
    #     hazard="Environmental Pollution",
    #     how_it_harms="Health impacts",
    #     who_it_harms="Local communities"
    # ),
    # Prevention(
    #     prevention="Geotechnical assessments",
    #     activity="Unregulated construction on hillsides",
    #     hazard="Landslide",
    #     how_it_harms="Structural damage",
    #     who_it_harms="Residents in affected areas"
    # ),
    # Prevention(
    #     prevention="International agreements for responsible space debris management",
    #     activity="Poorly managed space debris",
    #     hazard="Space Debris Impact",
    #     how_it_harms="Damage to satellites",
    #     who_it_harms="Space infrastructure"
    # ),
    # Prevention(
    #     prevention="Automated monitoring systems for real-time detection of potential anomalies in nuclear reactors",
    #     activity="Operating nuclear power plants",
    #     hazard="Nuclear Accident",
    #     how_it_harms="Radioactive contamination",
    #     who_it_harms="Workers at the plant"
    # )
]