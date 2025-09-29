import os
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.memory import ConversationBufferMemory

def create_csv_agent(df: pd.DataFrame, llm: ChatGoogleGenerativeAI, memory):
    """
    Cria um agente LangChain para interagir com um DataFrame do pandas.
    """
    agent_executor = create_pandas_dataframe_agent(
        llm=llm,
        df=df,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        memory=memory,
        handle_parsing_errors=True,
        allow_dangerous_code=True,
        max_iterations=5,
        early_stopping_method="generate"
    )
    return agent_executor