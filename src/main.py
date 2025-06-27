from agents import Runner, TResponseInputItem,Agent
from multi_agents.triage_agent import triage_agent
import asyncio
from config.model import config

async def start_chat(starting_agent: Agent, chat: list[TResponseInputItem]):
    print("NOTE: Chat started. Type 'EXIT' to end the conversation.")
    print("-----------------------------------------")

    current_agent = starting_agent  # 🔁 Start with topic agent

    while True:
        user_input = input("You: ")
        if user_input.strip().upper() == "EXIT":
            print("Agent: Goodbye!\n")
            break

        print("User:", user_input, "\n")

        chat.append({
            "content": user_input,
            "role": "user",
            "type": "message"
        })

        result = await Runner.run(
            starting_agent=current_agent,  # 🔁 Use last agent
            input=chat,
            run_config=config
        )

        chat.clear()
        chat.extend(result.to_input_list())

        # 🔁 Update current agent to last agent who responded
        current_agent = result.last_agent

        print(f"{result.last_agent.name}:", result.final_output, "\n", flush=True)


if __name__ == "__main__":
    chat_history: list[TResponseInputItem] = []
    asyncio.run(start_chat(starting_agent=triage_agent, chat=chat_history))
