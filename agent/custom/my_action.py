from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context
from utils import logger


@AgentServer.custom_action("my_action_111")
class MyCustomAction(CustomAction):

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> bool:

        logger.debug("my_action_111 is running!")

        return True
