import ast

from CommonFunctions import create_custom_logger, time_wrapper
import json
import requests
from time import sleep


class InformationExtractor:
    def __init__(self):
        self.url = ""
        self.platform_keys = ""
        self.auth_keys = ""
        self.local_logger = create_custom_logger(InformationExtractor.__name__)

    @time_wrapper
    def extract_information_from_each_conversation(self, conversation: str):

        payload = json.dumps(
            {
                "Conversation": conversation
            }
        )
        headers = {
            'AI-Service-Version': 'v0',
            'AI-Model': 'gpt-35-turbo',
            'Freddy-Ai-Platform-Authorization': self.platform_keys,
            'Authorization': self.auth_keys,
            'Content-Type': 'application/json'
        }
        response = requests.request("POST", self.url, headers=headers, data=payload)
        return response.text

    @time_wrapper
    def extract_information_from_all_conversations(self, conversation_list: list):
        res_list = []
        for ind_conversation in conversation_list:
            res_theme_sub_theme = self.extract_information_from_each_conversation(ind_conversation)
            res_theme_sub_theme = json.loads(res_theme_sub_theme)["content"].replace("'s ", "\'s ")
            res_theme_sub_theme = res_theme_sub_theme[res_theme_sub_theme.find("{"): res_theme_sub_theme.find("}") + 1]
            theme_sub_theme_dict = ast.literal_eval(res_theme_sub_theme)
            phrase_list = theme_sub_theme_dict["theme"] + theme_sub_theme_dict["sub_theme"]
            res_list.append(phrase_list)
            sleep(1)
        return res_list
