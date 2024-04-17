import ast

from .CommonFunctions import create_custom_logger, time_wrapper
import json
import requests
from time import sleep


class InformationExtractor:
    def __init__(self):
        self.url = "http://freddy-ai-platform-stage.freshedge.net/v1/ai-service/freshservice/azure/conversational_insights_generation"
        self.platform_keys = "ffa6551a-5889-490f-bfc9-13ffe75916f8-1682081771072-admin-key"
        self.auth_keys = "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJmcmVzaHNlcnZpY2UiLCJpYXQiOjE2ODE4ODcyMDZ9.DMmEND9hnKLp9QSsjPW4k3FE0jDLopMJNDd0MEFooVE"
        self.local_logger = create_custom_logger(InformationExtractor.__name__)

    @time_wrapper
    def extract_information_from_each_conversation(self, conversation: str):

        payload = json.dumps(
            {
                "conversation": conversation
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
