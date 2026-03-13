"""Google Sheets MCP Test and Analysis from Yahoo Finance."""
# Imports
import sys
import os

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from scripts import base_tools, utils

from langchain_mcp_adapters.client import MultiServerMCPClient
import asyncio

# Set UTF-8 encoding for Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

load_dotenv()

model = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="gpt-4.1-nano"
)

checkpointer = InMemorySaver()

GOOGLE_SHEETS_PROMPT = """You are a helpful Google Sheets assistant.

You have access to Google Sheets tools. When the user asks about spreadsheets:
- Use the list_spreadsheets tool to list all spreadsheets
- Use get_sheet_data to read sheet data
- Use create_spreadsheet to create new sheets

IMPORTANT: You MUST use the available tools to complete user requests. Do not try to answer without using tools."""

async def get_tools():
    mcp_config = utils.load_mcp_config("google-sheets", "yahoo-finance")
    client = MultiServerMCPClient(mcp_config)
    mcp_tools = await client.get_tools()

    tools = mcp_tools + [base_tools.web_search, base_tools.get_weather]

    problematic_tools = ['get_sheet_data', 'batch_update_cells', 'get_sheet_formulas', 'batch_update']
    
    safe_tools = [tool for tool in tools if tool.name not in problematic_tools]

    print(f"Loaded {len(safe_tools)} Tools")
    #print(f"Tools Available\n{[tool.name for tool in safe_tools]}")

    return safe_tools

async def google_sheet_agent(query, thread_id='default'):
    tools = await get_tools()

    config = {'configurable': {'thread_id': thread_id}}
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=GOOGLE_SHEETS_PROMPT,
        checkpointer=checkpointer
    )

    result = await agent.ainvoke(
        {'messages': [HumanMessage(query)]},
        config=config
    )

    response = result['messages'][-1].text

    print("\n============== Output =============")
    print(response)

async def ask():
    print("\nChat mode started. Type 'q' or 'quite' to exit.\n")
    while True:
        print("\n\n\nAsk Question. Type 'q' or 'quite' to exit.")
        query = input("You: ").strip()

        if query.lower() in ["q", "quite"]:
            print("Exiting chat mode.")
            break

        await google_sheet_agent(query)

if __name__ == "__main__":
    asyncio.run(ask())
