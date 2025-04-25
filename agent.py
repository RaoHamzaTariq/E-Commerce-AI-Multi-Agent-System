from agents import (
    Agent,
    RunContextWrapper,
    RunConfig,
    Runner,
    function_tool,
    OpenAIChatCompletionsModel,
    set_default_openai_client,
    set_tracing_disabled,
)

set_tracing_disabled(True)

from openai import AsyncOpenAI
import asyncio
from dotenv import load_dotenv
from typing import Optional
from dataclasses import dataclass
import os

from tools.tools import get_order_history, get_review_data, get_user_profile, save_product_recommendation, save_product_estimation, save_order_tracker, get_product_data, save_user_info
from data.dataModel import UserProfile, ProductEstimation, ProductRecommendation, OrderTracker, OrderDetail, UserInfo


# Load environment variables
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY is None:
    raise ValueError("GOOGLE_API_KEY environment variable is not set.")

# Set up the Gemini API-compatible client
client = AsyncOpenAI(
    api_key=GOOGLE_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=client
)

# Wrap into RunConfig
config = RunConfig(
    model=model,
    model_provider=client,
    tracing_disabled=True,
)

from agents import handoff
from agents.extensions import handoff_filters
from agents import Agent, Runner, TResponseInputItem
async def start_chat(primary_agent: Agent, chat: list[TResponseInputItem]):
   
    print("NOTE: Chat started. You can type 'EXIT' to exit the conversation.")
    print("-----------------------------------------")
    while True:

        user_input = input("You: ")
        print("User: ", user_input, "\n")
        
        if user_input == "EXIT":
            print("Fitness Assistant: Goodbye!", "\n")
            break
        
        chat.append({
            "content": user_input,
            "role" : "user",
            "type": "message"
        })
        
        result = await Runner.run(
            starting_agent=primary_agent, 
            input=chat
        )
        
        chat.clear()
        chat.extend(result.to_input_list())


        print("Nike Assistant:", result.final_output, "\n", flush=True)

MainAgent = Agent(
    name="Nike Assistant",
    model=model,
    instructions="Nike Assistant is a virtual assistant that helps users with product recommendations, finding product, order tracking, and user profile management and support chat of faqs and policies by using different agents. Don't do anything by yourself. Just handover to the respective agent.",
    tools=[
        get_user_profile,
        save_product_estimation,
        save_product_recommendation,
        save_order_tracker,
        get_order_history,
        get_product_data,
        save_user_info
    ]
)



# Product Finder Agent
ProductFinderAgent = Agent(
    name="Product Finder",
    instructions="You are a product finder agent. You help users find products based on their preferences.First get the information about the product save it in ProductEstimation using save_product_estimation and then find the product using tool get_product_data. and then provide the product details. After that handover back to the Nike Assistant to get the product recommendation.",
    model=model,
    handoffs=[handoff(
        agent=MainAgent,
        input_filter=handoff_filters.remove_all_tools
    )],
    tools=[
        get_product_data,
        save_product_estimation,
        save_product_recommendation,
        save_order_tracker,
        get_order_history, 
    ]
)

# Product Recommendation Agent
ProductRecommendationAgent = Agent(
    name="Product Recommendation",
    instructions="You are a product recommendation agent. You help users to suggest the product based on their preferences (Product Estimation) and then save it in ProductRecommendation using save_product_recommendation . then handoff to the main agent.",
    model=model,
    handoffs=[handoff(
        agent=MainAgent,
        input_filter=handoff_filters.remove_all_tools
    )],
    tools=[
        get_product_data,
        save_product_estimation,
        save_product_recommendation,
        save_order_tracker,
        get_order_history, 
    ]
)

# Order Tracker Agent
OrderTrackerAgent = Agent(
    name="Order Tracker",
    instructions="You are an order tracker agent. You help users to track their orders. First get the user email and then get the order history using tool get_order_history and save it in OrderTracker using save_order_tracker. After that handover back to the Nike Assistant.",
    model=model,
    handoffs=[handoff(
        agent=MainAgent,
        input_filter=handoff_filters.remove_all_tools
    )],
    tools=[
        get_order_history,
        save_order_tracker, 
    ]
)


#  Review Analysis Agent
ReviewAnalysisAgent = Agent(
    name="Review Analysis",
    instructions="You are a review analysis agent. You help users to get the reviews and ratings of the product. First get the product name and then get the review data using tool get_review_data.then handover back to the Nike Assistant.",
    model=model,
    handoffs=[handoff(
        agent=MainAgent,
        input_filter=handoff_filters.remove_all_tools
    )],
    tools=[
        get_review_data
    ]
)

# Support Chat Agent
SupportChatAgent = Agent(
    name="Support Chat",
    model=model,
    instructions="""You are a support chat agent. You help users about faqs of company.  After that handover back to the Nike Assistant.
    
    Common FAQs:
    1. What is the return policy?
    2. Where this is locally available?
    3. What payment methods are accepted?
    4. How do I contact customer service?
    5. What is the warranty policy?

    Common Responses:
    1. Our return policy allows you to return items within 30 days of purchase.
    2. You can find our products at only website.
    3. Now we accept only cash on delivery. But we are working on it to add more payment methods.
    4. You can contact customer service by email bistructure9211@gmail.com
    5. Our warranty policy covers defects in materials and workmanship for 1 month from the date of purchase.

    
    """,
    handoffs=[handoff(
        agent=MainAgent,
        input_filter=handoff_filters.remove_all_tools
    )]
)

MainAgent.handoffs = [
    handoff(OrderTrackerAgent),
    handoff(ProductFinderAgent),
    handoff(ProductRecommendationAgent),
    handoff(ReviewAnalysisAgent),
    handoff(SupportChatAgent)
]

chat = []
result = asyncio.run(start_chat(MainAgent, chat))