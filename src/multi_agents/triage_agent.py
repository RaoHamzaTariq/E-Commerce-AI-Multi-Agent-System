from src.config.model import model
from agents import Agent,Runner
from src.multi_agents.general_faqs_agent import general_faqs_agent
from src.multi_agents.order_placing_agent import order_placing_agent
from src.multi_agents.order_tracking_agent import order_tracking_agent
from src.multi_agents.product_recommendation_agent import product_recommendation_agent
import asyncio

triage_agent = Agent(
    name="Triage Agent",
    instructions="""
    You are the Triage Agent for an e-commerce website. Your role is to analyze the user's query and determine which specialized agent should handle it. You must quickly identify whether the user is asking about:

    1. Order tracking (route to Order Tracking Agent)
    2. Product recommendations (route to Product Recommendation Agent)
    3. Placing a new order (route to Order Placing Agent)
    4. General questions about the company, policies, or website (route to General Queries Agent)

    Do not attempt to answer the query yourself. Simply identify the most appropriate agent and handoff the query to that agent.

    For greeting, response politely without handoff
    """,
    model=model,
    handoffs=[
        product_recommendation_agent,
        order_tracking_agent,
        order_placing_agent,
        general_faqs_agent
    ]
)

