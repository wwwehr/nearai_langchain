# This is a somewhat simplified version of
# https://github.com/coinbase/agentkit/blob/main/python/examples/langchain-cdp-chatbot/chatbot.py
# that adds nearai integration: inference, environment, and threads.

from pdb import set_trace as bp
import argparse
import json
import os
import sys
from waterbear import Bear

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

from pydantic import BaseModel, Field
from nearai_langchain.orchestrator import NearAILangchainOrchestrator, RunMode

# Configure a file to persist the agent's CDP MPC Wallet Data.
wallet_data_file = "wallet_data.txt"

load_dotenv()

orchestrator = NearAILangchainOrchestrator(globals())
# To continue conversation on existing thread in local mode:
# orchestrator = NearAILangchainOrchestrator(globals(), thread_id="thread_xxxxxx")

try:
    with open("metadata.json", "r") as f:
        metadata = Bear(**json.load(f))
except FileNotFoundError:
    print("Error: File metadata.json not found")
except json.JSONDecodeError as e:
    print(f"Error: Invalid JSON in file: {e}")

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

def main(case_file_path: str) -> bool:
    """Start the chatbot agent."""

    try:
        with open(case_file_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {case_file_path} not found")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in file: {e}")

    TASK = f"""
    Your name is Doctor Feelgood. You're the one that makes you feel, alright.
    You are a helpful human physician with an expertise in Internal Medicine.
    You have been asked to help diagnose a patient. Here are all the case
    details.

    INTAKE:
    {json.dumps(data['INTAKE'])}

    CHART:
    {json.dumps(data['CHART'])}

    Form an opinion and suggest a diagnosis. Summarize your reasoning.
    """
    env.add_user_message(TASK)

    messages = env.list_messages()
    for chunk in executor.stream({"messages": messages}):
        if "agent" in chunk:
            result = chunk["agent"]["messages"][0].content
        elif "tools" in chunk:
            result = chunk["tools"]["messages"][0].content
        env.add_reply(result)

    print("-------------------")

    # Run once per user message.
    env.mark_done()

    model_name = metadata.details.agent.defaults.model
    TASK = f"""
    You are a doctor named Doctor McCoy working at the {model_name} institution.
    You have already formed an expert opinion and formed a diagnosis of a patients
    condition. This is your DIAGNOSTIC REPORT and this report is found below.

    You must upload your DIAGNOSTIC REPORT into the database using an existing schema that
    is in the Nillion SecretVault.

    STEPS:
    1. you must lookup the schema to use using the nillion_lookup_schema tool to
       find the `Medical Diagnostic Report` database.
    2. transform your DIAGNOSTIC REPORT to match the available fields in the schema.
    3. IMPORTANT: If you do not find an existing schema, do not create one, just stop.
    4. If you find a schema, you will use it's identifier, a UUID4, and your DIAGNOSTIC
       REPORT to upload to the database.
    5. Your job is to upload the DIAGNOSTIC REPORT and you need to kep trying until
       it is successful.

    YOUR DIAGNOSTIC REPORT IS:
    {result}
    """

    env.add_user_message(TASK)
    messages = env.list_messages()
    for chunk in executor.stream({"messages": messages}):
        if "agent" in chunk:
            result = chunk["agent"]["messages"][0].content
        elif "tools" in chunk:
            result = chunk["tools"]["messages"][0].content
        env.add_reply(result)
    print(result)
    env.mark_done()
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze input file and write case report to Nillion SecretVault"
    )
    parser.add_argument("case_file_path", type=str, help="Path to case file")
    args = parser.parse_args()
    print(f"The doctor is IN [{args.case_file_path}]...")
    main(args.case_file_path)
