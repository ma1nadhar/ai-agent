# Importing tools for web search and Wikipedia queries
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper

# Tool wrapper class from LangChain to define custom tools
from langchain.tools import Tool

# Used for adding timestamps to saved data
from datetime import datetime

# -------------------------------------------
# Function: Save research data to a text file
# -------------------------------------------
def save_to_txt(data: str, filename: str = "research_output.txt"):
    # Get the current date and time in a readable format
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Format the output with a header and timestamp
    formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"

    # Open the file in append mode and write the formatted data
    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)
    
    # Return confirmation message
    return f"Data successfully saved to {filename}"

# -------------------------------------------
# Tool 1: Saving output to a file
# -------------------------------------------
save_tool = Tool(
    name="save_text_to_file",          # Name of the tool
    func=save_to_txt,                  # Function to execute
    description="Saves structured research data to a text file.",  # Description for agent
)

# -------------------------------------------
# Tool 2: DuckDuckGo Web Search Tool
# -------------------------------------------
# Initialize DuckDuckGo search functionality
search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="search",                     # Name used by the agent
    func=search.run,                  # The function that executes the search
    description="Search the web for information",
)

# -------------------------------------------
# Tool 3: Wikipedia Search Tool
# -------------------------------------------
# Set up a wrapper to limit Wikipedia API results and content length
api_wrapper = WikipediaAPIWrapper(
    top_k_results=1,                  # Only get the top result
    doc_content_chars_max=100         # Limit content to first 100 characters
)

# Create the Wikipedia query tool using the configured wrapper
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)
