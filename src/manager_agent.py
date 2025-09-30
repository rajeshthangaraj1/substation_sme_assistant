# D:\rajesh\python\substation_SME_assistant\src\manager_agent.py

from src.chat_handler import ChatHandler
from src.utils.logger import get_logger

logger = get_logger()


class ManagerAgent:
    """
    Manager Agent:
    - Future-ready to handle multiple agents (RAG, MySQL, Summarizer, etc.)
    - Currently only one active agent: Document Q&A via ChatHandler
    """

    def __init__(self):
        self.doc_agent = ChatHandler()

    def route(self, user_query: str):
        """
        Route query to the appropriate agent.
        For now, always routes to document agent (ChatHandler).
        """
        logger.info(f"ManagerAgent routing query: {user_query}")
        return self.doc_agent.answer_question(user_query)
