import ast
import json

import agenthub.advanced_browsing_agent.dynamic_prompting as dp
from opendevin.controller.action_parser import ResponseParser
from opendevin.events.action import (
    Action,
    BrowseInteractiveAction,
)


class BrowsingResponseParser(ResponseParser):
    def __init__(self, main_prompt: dp.Shrinkable):
        # Need to pay attention to the item order in self.action_parsers
        super().__init__()
        self.main_prompt = main_prompt

    def parse_response(self, response: str) -> str:
        # add the stop token back if it is not empty
        return response

    def parse_action(self, action_str: str) -> Action:
        answer_dict = self.main_prompt.parse_answer(action_str)
        browser_action_str = answer_dict['action']
        answer_dict.pop('action')
        # dumb the answer dict to string
        thought_str = json.dumps(answer_dict)

        msg_content = ''
        # special care to send_msg_to_user since we need to convert it to MessageAction
        for sub_action in browser_action_str.split('\n'):
            if 'send_msg_to_user(' in sub_action:
                tree = ast.parse(sub_action)
                args = tree.body[0].value.args  # type: ignore
                msg_content = args[0].value

        return BrowseInteractiveAction(
            browser_actions=browser_action_str,
            thought=thought_str,
            browsergym_send_msg_to_user=msg_content,
        )

    def parse(self, response: str) -> Action:
        action_str = self.parse_response(response)
        action = self.parse_action(action_str)
        return action
