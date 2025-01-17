import os
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.prompts import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder,
    ChatPromptTemplate,
    PromptTemplate,
)
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.agents import create_tool_calling_agent, AgentExecutor
from dotenv import load_dotenv; load_dotenv()
from datetime import datetime

os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
# System Prompt Template
system_prompt_template = f"""You an educational assistant for students on any level, you are to assist them on all matters relating to academics, admission process,
scholarship processes and all other related matters like conselling and advising them, you cannot reply to requests that are not academically related, 
such as generating code. except it's academically related. Always use the search tool for the latest news and informations you have no access/answer to, generally.
Today's date is: {datetime.now().strftime("%m-%d-%Y")}
"""

# Prompt Setup
chat_prompts = [
    SystemMessagePromptTemplate(
        prompt=PromptTemplate(template=system_prompt_template)
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate(
        prompt=PromptTemplate(template="{input}", input_variables=["input"])
    ),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
]

prompt = ChatPromptTemplate(chat_prompts)
llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)

api_wrapper = DuckDuckGoSearchAPIWrapper()

search_desc = "search tool based on duckduckgo, useful when users have questions and you dont have answers to them, questions asking for latest info. input should be a search query."
search = DuckDuckGoSearchResults(api_wrapper=api_wrapper, source="news", description=search_desc, )
tools = [search]
agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)