import json
import os

from browsergym.core.action.highlevel import HighLevelActionSet
from browsergym.utils.obs import flatten_axtree_to_str

import agenthub.advanced_browsing_agent.dynamic_prompting as dp
from agenthub.advanced_browsing_agent.generic_agent_prompt import (
    GenericPromptFlags,
    MainPrompt,
)
from agenthub.advanced_browsing_agent.response_parser import BrowsingResponseParser
from opendevin.controller.agent import Agent
from opendevin.controller.state.state import State
from opendevin.core.logger import opendevin_logger as logger
from opendevin.events.action import (
    Action,
    AgentFinishAction,
    BrowseInteractiveAction,
)
from opendevin.events.observation import BrowserOutputObservation
from opendevin.events.observation.observation import Observation
from opendevin.llm.llm import LLM
from opendevin.runtime.plugins import (
    PluginRequirement,
)
from opendevin.runtime.tools import RuntimeTool

EVAL_DATASET = os.getenv('EVAL_DATASET', None)
# define a configurable action space, with chat functionality, web navigation, and webpage grounding using accessibility tree and HTML.
# see https://github.com/ServiceNow/BrowserGym/blob/main/core/src/browsergym/core/action/highlevel.py for more details
DISABLE_NAV = True if EVAL_DATASET in ['miniwob'] else False
WAIT_INITIAL_PAGE = True if EVAL_DATASET in ['webarena', 'miniwob'] else False
ACTION_SPACE = HighLevelActionSet(
    subsets=(['chat', 'bid'] if DISABLE_NAV else ['chat', 'bid', 'nav']),
    strict=False,  # less strict on the parsing of the actions
    multiaction=True,  # enable to agent to take multiple actions at once
)


# the configuration is taken from agentlab, check https://github.com/ServiceNow/AgentLab/blob/main/src/agentlab/agents/generic_agent/configs.py#L60 for more details
PROMPT_FLAGS = GenericPromptFlags(
    obs=dp.ObsFlags(
        use_html=False,
        use_ax_tree=True,
        use_focused_element=True,
        use_error_logs=True,
        use_history=True,
        use_past_error_logs=False,
        use_action_history=True,
        use_think_history=False,
        use_diff=False,
        html_type='pruned_html',  # doesn't matter if not use_html
        use_screenshot=False,
        use_som=False,
        extract_visible_tag=True,
        extract_clickable_tag=True,
        extract_coords='False',
        filter_visible_elements_only=False,
    ),
    action=dp.ActionFlags(
        multi_actions=False,
        action_set='bid',  # is overwritten by the ACTION_SPACE
        long_description=True,
        individual_examples=True,
    ),
    use_plan=False,
    use_criticise=False,
    use_thinking=True,
    use_memory=False,
    use_concrete_example=True,
    use_abstract_example=True,
    use_hints=True,
    enable_chat=False,
    max_prompt_tokens=None,
    be_cautious=True,
    extra_instructions=None,
)


def get_error_prefix(last_browser_action: str) -> str:
    return """IMPORTANT! Last action is incorrect:
{last_browser_action}
Think again with the current observation of the page."""


class BrowsingAgent(Agent):
    VERSION = '2.0'
    """
    An agent that interacts with the browser.
    Compared to the vanilla browsing agent, this version is more flexible in constructing the prompt
    """

    sandbox_plugins: list[PluginRequirement] = []
    runtime_tools: list[RuntimeTool] = [RuntimeTool.BROWSER]

    def __init__(
        self,
        llm: LLM,
    ) -> None:
        """
        Initializes a new instance of the BrowsingAgent class.

        Parameters:
        - llm (LLM): The llm to be used by this agent
        """
        super().__init__(llm)
        self.reset()

    def reset(self) -> None:
        """
        Resets the Browsing Agent.
        """
        super().reset()
        self.cost_accumulator = 0
        self.error_accumulator = 0

        self.plan = 'No plan yet'
        self.plan_step = -1
        self.thoughts: list[str] = []
        self.memories: list[str] = []
        self.actions: list[str] = []
        self.obs_history: list[Observation] = []

    def step(self, state: State) -> Action:
        """
        Performs one step using the Browsing Agent.
        This includes gathering information on previous steps and prompting the model to make a browsing command to execute.

        Parameters:
        - state (State): used to get updated info

        Returns:
        - BrowseInteractiveAction(browsergym_command) - BrowserGym commands to run
        - MessageAction(content) - Message action to run (e.g. ask for clarification)
        - AgentFinishAction() - end the interaction
        """

        obs_history: list[Observation] = []

        if WAIT_INITIAL_PAGE and len(state.history) == 1:
            # retrieve the initial observation already in browser env
            # initialize and retrieve the first observation by issuing an noop OP
            # For non-benchmark browsing, the browser env starts with a blank page, and the agent is expected to first navigate to desired websites
            return BrowseInteractiveAction(browser_actions='noop()')

        # get the overall goal
        if (goal := state.get_current_user_intent()) is None:
            goal = state.inputs['task']

        # add the last action and observation to the history
        last_action, last_obs = state.history[-1]
        # no interactions allowed with the user
        # TODO [shuyanzh] extend to be more flexible
        assert isinstance(last_action, BrowseInteractiveAction)
        assert isinstance(last_obs, BrowserOutputObservation)

        if last_action.browser_actions != 'noop()':
            self.actions.append(last_action.browser_actions)

        # action.thought is a dictionary of multiple components
        thoughts_dict = json.loads(last_action.thought)
        self.thoughts.append(thoughts_dict.get('think', None))
        self.memories.append(thoughts_dict.get('memory', None))
        self.plan = thoughts_dict.get('plan', self.plan)
        self.plan_step = thoughts_dict.get('step', self.plan_step)

        # add axtree str to last obs
        try:
            axtree_txt = flatten_axtree_to_str(
                last_obs.axtree_object,
                extra_properties=last_obs.extra_element_properties,
                with_visible=PROMPT_FLAGS.obs.extract_visible_tag,
                with_clickable=PROMPT_FLAGS.obs.extract_clickable_tag,
                with_center_coords=PROMPT_FLAGS.obs.extract_coords == 'center',
                with_bounding_box_coords=PROMPT_FLAGS.obs.extract_coords == 'box',
                filter_visible_only=PROMPT_FLAGS.obs.filter_visible_elements_only,
                filter_with_bid_only=PROMPT_FLAGS.obs.filter_with_bid_only,
                filter_som_only=PROMPT_FLAGS.obs.filter_som_only,
            )
            setattr(last_obs, 'axtree_txt', axtree_txt)

        except Exception as e:
            logger.error('Error when trying to process the accessibility tree: %s', e)
            return AgentFinishAction(
                outputs={'content': 'Error encountered when browsing.'}
            )

        # if the final BrowserInteractiveAction contains send_msg_to_user,
        # we should also send a message back to the user in OpenDevin and call it a day
        if last_action.browsergym_send_msg_to_user:
            return AgentFinishAction(
                outputs={'content': last_action.browsergym_send_msg_to_user}
            )

        messages = []
        messages.append({'role': 'system', 'content': dp.SystemPrompt().prompt})
        # construct the user message
        main_prompt = MainPrompt(
            action_set=ACTION_SPACE,
            goal=goal,
            obs_history=obs_history,
            actions=self.actions,
            memories=self.memories,
            thoughts=self.thoughts,
            previous_plan=self.plan,
            step=self.plan_step,
            flags=PROMPT_FLAGS,
        )
        # shrink the prompt to fit the tokens
        prompt = dp.fit_tokens(
            main_prompt,
            model_name=(
                'openai/gpt-4'
                if 'gpt-4' in self.llm.model_name
                else self.llm.model_name
            ),
        )
        messages.append({'role': 'user', 'content': prompt})
        logger.info(prompt)

        response = self.llm.completion(
            messages=messages,
            temperature=0.0,
        )['choices'][0]['message']['content'].strip()
        response_parser = BrowsingResponseParser(main_prompt)
        return response_parser.parse(response)

    def search_memory(self, query: str) -> list[str]:
        raise NotImplementedError('Implement this abstract method')
