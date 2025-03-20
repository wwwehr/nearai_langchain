# This is a somewhat simplified version of
# https://github.com/coinbase/agentkit/blob/main/python/examples/langchain-cdp-chatbot/chatbot.py
# that adds nearai integration: inference, environment, and threads.

import json
import os

from coinbase_agentkit import (  # type: ignore
    AgentKit,
    AgentKitConfig,
    CdpWalletProvider,
    CdpWalletProviderConfig,
    cdp_api_action_provider,
    cdp_wallet_action_provider,
    erc20_action_provider,
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

    cdp_config = None
    if wallet_data is not None:
        cdp_config = CdpWalletProviderConfig(wallet_data=wallet_data)

    wallet_provider = CdpWalletProvider(cdp_config)

    agentkit = AgentKit(
        AgentKitConfig(
            wallet_provider=wallet_provider,
            action_providers=[
                cdp_api_action_provider(),
                cdp_wallet_action_provider(),
                erc20_action_provider(),
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

    risk_profiles = {
        "Conservative Trader": """
            Account Risk Tolerance: 1-2%
            Preferred Risk Categories: Very Low to Low
            Position Sizing: 5-15% max per asset
            Stop Loss Strategy: Wider (8-10%)
            Take Profit: 15-20%
            Time Horizon: Medium to Long-term
        """,
        "Balanced Trader": """
            Account Risk Tolerance: 2-5%
            Preferred Risk Categories: Low to Moderate
            Position Sizing: 10-20% max per asset
            Stop Loss Strategy: Standard (6-8%)
            Take Profit: 20-30%
            Time Horizon: Medium-term
        """,
        "Aggressive Trader": """
            Account Risk Tolerance: 5-10%
            Preferred Risk Categories: Moderate to High
            Position Sizing: 15-25% max per asset
            Stop Loss Strategy: Tighter (4-6%)
            Take Profit: 25-40%
            Time Horizon: Short to Medium-term
        """,
        "Day Trader": """
            Account Risk Tolerance: 3-7%
            Preferred Risk Categories: Any (adjusts position size accordingly)
            Position Sizing: Varies by volatility (5-30%)
            Stop Loss Strategy: Very tight (2-4%)
            Take Profit: 10-15%
            Time Horizon: Intraday to 48 hours
        """,
    }

    TASK = """
    Create an entry in the SecretVault that defines my risk trading strategy.

    First, look up the schema `My Crypto Trader Risk Profile` and consider the fields.

    Then, create a single entry that considers my trading style:
    {risk_profiles['Aggressive Trader']}
    """

    env.add_user_message(TASK)


messages = env.list_messages()
for chunk in executor.stream({"messages": messages}):
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
