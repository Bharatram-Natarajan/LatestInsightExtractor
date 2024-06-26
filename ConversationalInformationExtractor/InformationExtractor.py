import ast

from .CommonFunctions import create_custom_logger, time_wrapper
import json
import requests
from time import sleep
from tqdm.notebook import tqdm

class InformationExtractor:
    def __init__(self):
        # self.url = "http://freddy-ai-platform-stage.freshedge.net/v1/ai-service/freshservice/azure/conversational_insights_generation"
        self.url = "http://freddy-ai-platform-stage.freshedge.net/v1/ai-service/freshservice/azure/conversational_detailed_insights_generation"
        self.platform_keys = "ffa6551a-5889-490f-bfc9-13ffe75916f8-1682081771072-admin-key"
        self.auth_keys = "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJmcmVzaHNlcnZpY2UiLCJpYXQiOjE2ODE4ODcyMDZ9.DMmEND9hnKLp9QSsjPW4k3FE0jDLopMJNDd0MEFooVE"
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
            # 'AI-Model': 'gpt-35-turbo',
            'AI-Model': 'gpt-35-turbo-1106',
            'Freddy-Ai-Platform-Authorization': self.platform_keys,
            'Authorization': self.auth_keys,
            'Content-Type': 'application/json'
        }
        response = requests.request("POST", self.url, headers=headers, data=payload)
        return response.text

    # Function for themes and sub-themes only
    # @time_wrapper
    # def extract_information_from_all_conversations(self, conversation_list: list):
    #     res_list = []
    #     for ind_conversation in conversation_list:
    #         res_theme_sub_theme = self.extract_information_from_each_conversation(ind_conversation)
    #         res_theme_sub_theme = json.loads(res_theme_sub_theme)["content"].replace("'s ", "\'s ")
    #         res_theme_sub_theme = res_theme_sub_theme[res_theme_sub_theme.find("{"): res_theme_sub_theme.find("}") + 1]
    #         theme_sub_theme_dict = ast.literal_eval(res_theme_sub_theme)
    #         phrase_list = theme_sub_theme_dict["theme"] + theme_sub_theme_dict["sub_theme"]
    #         res_list.append(phrase_list)
    #         sleep(1)
    #     return res_list

    @time_wrapper
    def extract_information_from_all_conversations(self, conversation_list: list):
        ind_res_list, combined_res_list, combined_theme_sub_theme_list = [], [], []
        for ind_conversation in tqdm(conversation_list, desc='Conversation Progress'):
            insight_results = self.extract_information_from_each_conversation(ind_conversation)
            try:
                final_res = json.loads(insight_results)["content"]
                insights_results_dict = json.loads(final_res[final_res.find("{"): final_res.rfind("}") + 1])
                # print(insights_results_dict)
                phrase_list = []
                phrase_list.extend(insights_results_dict["important_topics"])
                for theme_info, sub_theme_info in insights_results_dict["important_themes_and_sub-themes"].items():
                    phrase_list.append(theme_info)
                    phrase_list.extend(sub_theme_info)
                # phrase_list = insights_results_dict["important_topics"] + insights_results_dict["important_themes"] + \
                #     insights_results_dict["important_sub-themes"] + insights_results_dict["main_entities"]
                combined_theme_sub_theme_list.append(insights_results_dict["important_themes_and_sub-themes"])
            except Exception as e:
                print(e)
                phrase_list = []
                insights_results_dict = final_res
                combined_theme_sub_theme_list.append({})
            combined_res_list.append(phrase_list)
            ind_res_list.append(insights_results_dict)
            sleep(1)
        return ind_res_list, combined_res_list, combined_theme_sub_theme_list
