from .CommonFunctions import create_custom_logger, time_wrapper
from .InformationCalculator import InformationCalculator
from .InformationExtractor import InformationExtractor


class ConversationalImportantPhraseGenerator:
    def __init__(self):
        self.local_logger = create_custom_logger(ConversationalImportantPhraseGenerator.__name__)
        self.extractor = InformationExtractor()
        self.calculator = InformationCalculator()

    @time_wrapper
    def process_conversations(self, conversation_list: list):
        self.local_logger.info("Calling chatgpt in bulk to generate important phrases with some sleep")
        info_list = self.extractor.extract_information_from_all_conversations(conversation_list)
        ind_insight_details, imp_phrase_list= info_list
        self.local_logger.info("Generate top 10 phrases from all the conversations")
        final_phrase_list = self.calculator.generate_and_sort_probability_for_each_type(imp_phrase_list)
        return ind_insight_details, final_phrase_list

