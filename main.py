# Load environment variables from a .env file (this is used to securely store API keys, etc.)
from dotenv import load_dotenv

# Pydantic is used to define structured data models and validate output
from pydantic import BaseModel

# Import Large Language Model interfaces from LangChain (Anthropic + OpenAI support)
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# Used to structure prompts for chat-based LLMs
from langchain_core.prompts import ChatPromptTemplate

# Parses model output into a Pydantic model (for structured responses)
from langchain_core.output_parsers import PydanticOutputParser

# Tools that agents can call, and utilities to build an agent
from langchain.agents import create_tool_calling_agent, AgentExecutor

# Custom tools you've created/imported for use in the agent (search, Wikipedia, saving files, etc.)
from tools import search_tool, wiki_tool, save_tool

# Load .env file (loads environment variables like API keys into the environment)
load_dotenv()

# Define the expected response format using Pydantic
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

# Initialize the LLM (Using Anthropic's Claude 3.5 Sonnet)
llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")

# Initialize the output parser to enforce the structure of the ResearchResponse
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# Define the prompt template for the agent
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use necessary tools. 
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        # Placeholder for previous conversation history
        ("placeholder", "{chat_history}"),
        # The user's question
        ("human", "{query}"),
        # Placeholder where the agent thinks and decides which tools to call
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())  # Adds instructions about output format

# List of tools the agent is allowed to use
tools = [search_tool, wiki_tool, save_tool]

# Create the agent that can call tools and use the provided LLM and prompt
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

# Create an executor to run the agent with verbose output (shows internal steps)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Ask the user for a research query
query = input("What can I help you research? ")

# Run the agent with the user's query
raw_response = agent_executor.invoke({"query": query})

# Try to parse the output into the structured ResearchResponse format
try:
    # Some agents return output inside a list, so we extract the text properly
    structured_response = parser.parse(raw_response.get("output")[0]["text"])
    print(structured_response)
except Exception as e:
    # If parsing fails, show the error and raw response to debug
    print("Error parsing response", e, "Raw Response - ", raw_response)
