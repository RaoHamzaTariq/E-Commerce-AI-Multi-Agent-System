from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel,RunConfig,set_default_openai_client,set_tracing_disabled
import os
from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY=os.getenv("GOOGLE_API_KEY")

external_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

set_tracing_disabled(True)
set_default_openai_client(client=external_client)