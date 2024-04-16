# Conversational Information Extractor

The package generates important phrases on analysing several conversations using LLMs of user choice.These phrases 
aides in understanding the trend of the past conversations for the company to plan their future goals or gain insights.
Additionally these phrases provide extra information that can be used in downstream task.

## Running the package
Follow the below steps in order to use this package.
1. Import the package
        ```
        from ConversationalInformationExtractor.ConversationalImportantPhraseGenerator import ConversationalImportantPhraseGenerator
        ```
2. Create an instance of the class
        ```
        important_phrase_extractor = ConversationalImportantPhraseGenerator()
        ```
3. Call the main function for processing the list of conversations
        ```
        res = important_phrase_extractor(<list of conversations where each conversation can be a list or single string.>)
        ```
4. The result of the call would provide the important phrases from all the list of conversations.

