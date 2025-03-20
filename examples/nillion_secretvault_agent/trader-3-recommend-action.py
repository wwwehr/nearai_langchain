# This is a somewhat simplified version of
# https://github.com/coinbase/agentkit/blob/main/python/examples/langchain-cdp-chatbot/chatbot.py
# that adds nearai integration: inference, environment, and threads.


import json
import os
import sys

from coinbase_agentkit import (  # type: ignore
    AgentKit,
    AgentKitConfig,
    CdpWalletProvider,
    CdpWalletProviderConfig,
    cdp_api_action_provider,
    cdp_wallet_action_provider,
    nillion_action_provider,
    pyth_action_provider,
    wallet_action_provider,
    weth_action_provider,
)
from coinbase_agentkit_langchain import get_langchain_tools  # type: ignore
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent

from nearai_langchain.orchestrator import NearAILangchainOrchestrator, RunMode

# Configure a file to persist the agent's CDP MPC Wallet Data.
wallet_data_file = "wallet_data.txt"

load_dotenv()

orchestrator = NearAILangchainOrchestrator(globals())
# To continue conversation on existing thread in local mode:
# orchestrator = NearAILangchainOrchestrator(globals(), thread_id="thread_xxxxxx")


def initialize_agent():
    """Initialize the agent with CDP Agentkit."""
    # Get ChatOpenAI model.
    llm = orchestrator.chat_model.chat_open_ai_model

    # Initialize CDP Wallet Provider
    wallet_data = None
    if os.path.exists(wallet_data_file):
        with open(wallet_data_file) as f:
            wallet_data = f.read()

    cdp_config = CdpWalletProviderConfig(network_id=os.environ["NETWORK_ID"])
    if wallet_data is not None:
        cdp_config = CdpWalletProviderConfig(
            network_id=os.environ["NETWORK_ID"], wallet_data=wallet_data
        )

    wallet_provider = CdpWalletProvider(cdp_config)

    agentkit = AgentKit(
        AgentKitConfig(
            wallet_provider=wallet_provider,
            action_providers=[
                cdp_api_action_provider(),
                cdp_wallet_action_provider(),
                nillion_action_provider(),
                pyth_action_provider(),
                wallet_action_provider(),
                weth_action_provider(),
            ],
        )
    )

    wallet_data_json = json.dumps(wallet_provider.export_wallet().to_dict())

    with open(wallet_data_file, "w") as f:
        f.write(wallet_data_json)

    # use get_langchain_tools
    tools = get_langchain_tools(agentkit)

    # Create ReAct Agent using the LLM and CDP Agentkit tools.
    return create_react_agent(
        llm,
        tools=tools,
        state_modifier=(
            "You are a helpful agent that can interact onchain using the Coinbase Developer Platform AgentKit. "
            "You are empowered to interact onchain using your tools. If you ever need funds, you can request "
            "them from the faucet if you are on network ID 'base-sepolia'. If not, you can provide your wallet "
            "details and request funds from the user. Before executing your first action, get the wallet details "
            "to see what network you're on. If there is a 5XX (internal) HTTP error code, ask the user to try "
            "again later. If someone asks you to do something you can't do with your currently available tools, "
            "you must say so, and encourage them to implement it themselves using the CDP SDK + Agentkit, "
            "recommend they go to docs.cdp.coinbase.com for more information. Be concise and helpful with your "
            "responses. Refrain from restating your tools' descriptions unless it is explicitly requested."
        ),
    )


executor = initialize_agent()

# NEAR AI environment.
# In remote mode thread is assigned, user messages are given, and an agent is run at least once per user message.
# In local mode an agent is responsible to get and upload user messages.
env = orchestrator.env

if orchestrator.run_mode == RunMode.LOCAL:
    print("Entering autonomous mode...")

    with open("supported-tokens.json", "r", encoding="utf-8") as f:
        supported_tokens = json.load(f)

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        report1 = f.read()

    with open(sys.argv[2], "r", encoding="utf-8") as f:
        report2 = f.read()

    TASK = f"""
    ## Your task:
    Perform zero or more swaps based on my risk profile and the prepared market analysis report.

    1. fetch my data in the `My Crypto Trader Risk Profile` schema and consider
    my settings. You should look up the schema based on the name.

    2. look up my wallet details and all held assets

    3. considering my assets and capital, recommend three (3) most impactful trades

    4. decide on smart risk trading strategy derived from trends discovered among the
       two reports and select the corresponding trading rules from my profile

    5. instead of looking up prices, just swap

    6. perform the swaps on cdp that you ultimately find most excellent and aligned
       with my risk profile without asking, but show me all the transaction ids

    7. tell me what actions are are performing before you perform them, summarize the
       result of each action

    IMPORTANT:
    - Do not recommend generic token names, types or descriptors. You must recommend
      explicit crypto ticker symbols by name only combined with amount, like "BTC|0.000020"
      or "XRP|0.10" or "ETH|0.002"
    - Use entire wallet funds as my working capital, but obey all the risk profile rules
    - Embrace the trader risk profile fully, but avoid small trade amounts where gas fees would
      cost more than the asset
    - Never swap a token to the same token
    - Only consider tokens that exist in the `Supported Tokens` list. Do not even concern
      yourself with other assets.
    - All held assets are subject to swap
    - Avoid WETH

    Here are the market reports to base your recommendation on:

    ## Report 1:
    {report1}

    ----
    ## Report 2:
    {report2}

    ----

    ## Supported Tokens:
    {supported_tokens}
    """

    env.add_user_message(TASK)


messages = env.list_messages()
for chunk in executor.stream({"messages": messages}, {"recursion_limit": 1000}):
    if "agent" in chunk:
        result = chunk["agent"]["messages"][0].content
    elif "tools" in chunk:
        result = chunk["tools"]["messages"][0].content
    env.add_reply(result)

    if orchestrator.run_mode == RunMode.LOCAL:
        print(result)
        print("-------------------")

# Run once per user message.
env.mark_done()

env.add_user_message("complete any outstanding tasks")
messages = env.list_messages()
for chunk in executor.stream({"messages": messages}, {"recursion_limit": 1000}):
    if "agent" in chunk:
        result = chunk["agent"]["messages"][0].content
    elif "tools" in chunk:
        result = chunk["tools"]["messages"][0].content
    env.add_reply(result)

    if orchestrator.run_mode == RunMode.LOCAL:
        print(result)
        print("-------------------")

# Run once per user message.
env.mark_done()
