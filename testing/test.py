from dotenv import load_dotenv
from agent import RunConfig, OpenAIChatCompletionsModel, set_tracing_disabled
from openai import AsyncOpenAI
import os
import json
import asyncio
import pandas as pd
import requests
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, ConfigDict, Field
from agent import Agent, FunctionTool, RunContextWrapper, function_tool, Runner, TResponseInputItem, handoff
from agents.extensions import handoff_filters
import nest_asyncio

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Get API key from environment
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY is None:
    raise ValueError("GOOGLE_API_KEY environment variable is not set.")

# Set up the Gemini API-compatible client
client = AsyncOpenAI(
    api_key=GOOGLE_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Configure the model
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

set_tracing_disabled(True)



# Define data models for user profile
class UserInfo(BaseModel):
    name: str 
    age: int
    email: str

class ProductEstimation(BaseModel):
    product_name: str 
    product_type: str 
    price_range: int

class ProductRecommendation(BaseModel):
    product_name: str
    product_type: str
    product_price: int
    product_slug: str

class OrderDetail(BaseModel):
    order_id: str
    order_date: str
    order_status: str
    estimated_delivery_date: str
    no_of_products: int
    total_price: int
    products: List[Dict[str, Any]]

class OrderTracker(BaseModel):
    total_orders: int
    orders_detail: List[OrderDetail]

class UserProfile(BaseModel):
    user_info: UserInfo
    product_estimation: ProductEstimation
    product_recommendation: ProductRecommendation
    order_tracker: OrderTracker

    model_config = ConfigDict(arbitrary_types_allowed=True)

# Function tools for the agents
@function_tool
async def get_user_profile(wrapper: RunContextWrapper[UserProfile]) -> str:
    """
    Get user profile information.
    
    Returns:
        str: JSON representation of the user profile.
    """
    try:
        # Extract user profile information in a readable format
        user_info = wrapper.context.user_info
        profile_data = {
            "User Info": {
                "Name": user_info.name or "Not set",
                "Age": user_info.age or "Not set",
                "Email": user_info.email or "Not set"
            },
            "Product Interest": {
                "Product Type": wrapper.context.product_estimation.product_type or "Not set",
                "Product Name": wrapper.context.product_estimation.product_name or "Not set",
                "Price Range": wrapper.context.product_estimation.price_range or "Not set"
            },
            "Recommended Product": {
                "Product Name": wrapper.context.product_recommendation.product_name or "Not set",
                "Product Type": wrapper.context.product_recommendation.product_type or "Not set",
                "Product Price": wrapper.context.product_recommendation.product_price or "Not set"
            },
            "Order Information": {
                "Total Orders": wrapper.context.order_tracker.total_orders or "No orders",
                "Order Details": len(wrapper.context.order_tracker.orders_detail) or "No order details"
            }
        }
        return json.dumps(profile_data, indent=2)
    except Exception as e:
        return f"Failed to retrieve user profile: {str(e)}"

@function_tool
async def save_product_estimation(
    wrapper: RunContextWrapper[UserProfile],
    product_name: str, 
    product_type: str, 
    price_range: int
) -> str:
    """
    Save product estimation information of user.
    
    Args:
        product_name (str): Name of the product.
        product_type (str): Type of the product (e.g., "Running Shoes", "Sportswear").
        price_range (int): Price range of the product in dollars.
    
    Returns:
        str: Confirmation message with saved details.
    """
    try:
        wrapper.context.product_estimation = ProductEstimation(
            product_name=product_name,
            product_type=product_type,
            price_range=price_range
        )
        return f"Successfully saved product estimation: {product_name} ({product_type}) at price range ${price_range}"
    except Exception as e:
        return f"Failed to save product estimation: {str(e)}"

@function_tool
async def save_product_recommendation(
    wrapper: RunContextWrapper[UserProfile],
    product_name: str, 
    product_type: str, 
    product_price: int, 
    product_slug: str
) -> str:
    """
    Save product recommendation information for the user.
    
    Args:
        product_name (str): Name of the recommended product.
        product_type (str): Type of the product (e.g., "Running Shoes").
        product_price (int): Price of the product in dollars.
        product_slug (str): URL slug for the product.
    
    Returns:
        str: Confirmation message with recommended product details.
    """
    try:
        wrapper.context.product_recommendation = ProductRecommendation(
            product_name=product_name,
            product_type=product_type,
            product_price=product_price,
            product_slug=product_slug
        )
        return f"Successfully saved product recommendation: {product_name} ({product_type}) at ${product_price}"
    except Exception as e:
        return f"Failed to save product recommendation: {str(e)}"

@function_tool
async def save_order_tracker(
    wrapper: RunContextWrapper[UserProfile],
    total_orders: int, 
    orders_detail: List[Dict[str, Any]]
) -> str:
    """
    Save order tracker information for the user.
    
    Args:
        total_orders (int): Total number of orders.
        orders_detail (List[Dict]): List of order details including order status, 
                                   delivery date, number of products, and total price.
    
    Returns:
        str: Confirmation message with order tracking details.
    """
    try:
        # Convert the dictionary to OrderDetail objects
        order_details_list = []
        for order in orders_detail:
            order_detail = OrderDetail(
                order_id=order.get("order_id", ""),
                order_date=order.get("order_date", ""),
                order_status=order.get("order_status", ""),
                estimated_delivery_date=order.get("estimated_delivery_date", ""),
                no_of_products=order.get("no_of_products", 0),
                total_price=order.get("total_price", 0),
                products=order.get("products", [])
            )
            order_details_list.append(order_detail)
        
        wrapper.context.order_tracker = OrderTracker(
            total_orders=total_orders,
            orders_detail=order_details_list
        )
        return f"Successfully saved order tracking information: {total_orders} orders"
    except Exception as e:
        return f"Failed to save order tracker: {str(e)}"

@function_tool
async def get_order_history(
    wrapper: RunContextWrapper[UserProfile], 
    user_email: str
) -> str:
    """
    Retrieves order history data for a given user email from the API.

    Args:
        user_email (str): The email address of the user.

    Returns:
        str: Formatted order history or error message.
    """
    try:
        # Make API request
        response = requests.get(f"http://localhost:3000/api/users?email={user_email}")
        response.raise_for_status()
        
        data = response.json().get('data', {})
        
        if not data or 'orderHistory' not in data or not data['orderHistory']:
            return "No order history found for this email address."
        
        # Create a structured DataFrame
        order_data = []
        for order in data.get('orderHistory', []):
            for item in order.get('productDetails', []):
                product = item.get('product_id', {})
                order_data.append({
                    "Order Date": pd.to_datetime(order.get('orderDate')).strftime('%Y-%m-%d') if order.get('orderDate') else "Unknown",
                    "Product": product.get('productName', "Unknown"),
                    "Price": f"${product.get('price', 0)}",
                    "Quantity": item.get('quantity', 0),
                    "Subtotal": f"${item.get('subtotal', 0)}",
                    "Payment Status": order.get('paymentStatus', "Unknown"),
                    "Order Status": order.get('orderStatus', "Unknown")
                })
        
        # Convert to DataFrame and format
        if order_data:
            df = pd.DataFrame(order_data)
            
            # Save to the user context
            orders_list = []
            order_groups = df.groupby("Order Date")
            
            total_orders = len(order_groups)
            
            for order_date, group in order_groups:
                products = []
                for _, row in group.iterrows():
                    products.append({
                        "name": row["Product"],
                        "price": row["Price"].replace("$", ""),
                        "quantity": row["Quantity"],
                        "subtotal": row["Subtotal"].replace("$", "")
                    })
                
                order_detail = {
                    "order_id": f"ord-{len(orders_list) + 1}",
                    "order_date": order_date,
                    "order_status": group["Order Status"].iloc[0],
                    "estimated_delivery_date": pd.to_datetime(order_date) + pd.Timedelta(days=5),
                    "no_of_products": len(products),
                    "total_price": sum([int(p["subtotal"]) for p in products]),
                    "products": products
                }
                orders_list.append(order_detail)
            
            # Save order information
            await save_order_tracker(wrapper, total_orders, orders_list)
            
            # Return formatted table
            return f"Found {total_orders} orders:\n\n{df.to_string(index=False)}"
        else:
            return "No order details found in the order history."
    
    except requests.exceptions.RequestException as e:
        return f"Error fetching order history: {str(e)}"
    except Exception as e:
        return f"Unexpected error retrieving order history: {str(e)}"

@function_tool
async def get_product_data(
    wrapper: RunContextWrapper[UserProfile],
    search_term: Optional[str] = None,
    category: Optional[str] = None,
    min_price: Optional[int] = None,
    max_price: Optional[int] = None
) -> str:
    """
    Fetches product data from the Nike API with optional filtering parameters.
    
    Args:
        search_term (str, optional): Term to search for in product names
        category (str, optional): Category to filter by (e.g., "Running", "Basketball")
        min_price (int, optional): Minimum price for filtering
        max_price (int, optional): Maximum price for filtering
        
    Returns:
        str: Formatted product data or error message
    """
    try:
        # Make API request
        response = requests.get("https://nike-marketplace-bi-structure.vercel.app/api/products")
        response.raise_for_status()
        
        data = response.json().get("data", [])
        if not data:
            return "No products found in the database."
        
        # Process products
        products = []
        for item in data:
            # Apply filters if provided
            name = item.get("productName", "")
            price = item.get("price", 0)
            cat = item.get("category", "")
            
            # Skip if doesn't match filters
            if search_term and search_term.lower() not in name.lower():
                continue
            if category and category.lower() not in cat.lower():
                continue
            if min_price is not None and price < min_price:
                continue
            if max_price is not None and price > max_price:
                continue
            
            # Add to results
            products.append({
                "Product Name": name,
                "Price": f"${price}",
                "Category": cat,
                "Colors": ", ".join(item.get("colors", [])),
                "Inventory": item.get("inventory", 0),
                "Status": item.get("status", ""),
                "Slug": item.get("slug", {}).get("current", "")
            })

        # Convert to DataFrame and format output
        if products:
            df = pd.DataFrame(products)
            
            # Limit to the most relevant columns for display
            display_columns = ["Product Name", "Price", "Category", "Colors", "Status"]
            display_df = df[display_columns]
            
            # Format the output
            result = f"Found {len(products)} products"
            if search_term:
                result += f" matching '{search_term}'"
            if category:
                result += f" in category '{category}'"
            if min_price is not None or max_price is not None:
                price_range = ""
                if min_price is not None:
                    price_range += f"${min_price}"
                price_range += " to "
                if max_price is not None:
                    price_range += f"${max_price}"
                result += f" within price range {price_range}"
                
            result += ":\n\n"
            result += display_df.to_string(index=False)
            
            # Add complete data as an attachment if we have a lot of products
            if len(products) > 10:
                result += "\n\n(Showing first 10 products only. Use more specific filters for targeted results.)"
            
            return result
        else:
            return "No products found matching your criteria."
    
    except requests.exceptions.RequestException as e:
        return f"Error fetching product data: {str(e)}"
    except Exception as e:
        return f"Unexpected error retrieving products: {str(e)}"

@function_tool
async def save_user_info(
    wrapper: RunContextWrapper[UserProfile], 
    name: str, 
    age: int, 
    email: str
) -> str:
    """
    Save basic user information to the user profile.
    
    Args:
        name (str): Name of the user.
        age (int): Age of the user.
        email (str): Email address of the user.
    
    Returns:
        str: Confirmation message.
    """
    try:
        wrapper.context.user_info = UserInfo(
            name=name,
            age=age,
            email=email
        )
        return f"Successfully saved user information for {name} (email: {email})"
    except Exception as e:
        return f"Failed to save user information: {str(e)}"

@function_tool
async def get_review_data(
    wrapper: RunContextWrapper[UserProfile],
    product_name: str
) -> str:
    """
    Retrieves review data (comments and ratings) for a given product.
    
    Args:
        product_name (str): The name of the product to fetch reviews for.
    
    Returns:
        str: Formatted review data or error message.
    """
    try:
        # Create slug from product name
        list_of_words = product_name.split(" ")
        lower_list = [part.lower() for part in list_of_words]
        slug = "-".join(lower_list)
        slug = slug.replace("'", "")

        # Make API request
        response = requests.get(f"https://nike-marketplace-bi-structure.vercel.app/api/products?slug={slug}")
        response.raise_for_status()
        
        data = response.json()
        if "data" in data and "reviews" in data["data"] and data["data"]["reviews"]:
            reviews_data = data["data"]["reviews"]
            
            # Create a structured output
            result = f"Reviews for {product_name}:\n\n"
            
            # Calculate average rating
            ratings = [review.get("rating", 0) for review in reviews_data]
            avg_rating = sum(ratings) / len(ratings) if ratings else 0
            
            result += f"Average Rating: {avg_rating:.1f}/5 ({len(ratings)} reviews)\n\n"
            
            # Format individual reviews
            for i, review in enumerate(reviews_data, 1):
                result += f"Review #{i}:\n"
                result += f"Rating: {'★' * int(review.get('rating', 0))}{' ' * (5 - int(review.get('rating', 0)))}\n"
                result += f"Comment: {review.get('comment', 'No comment')}\n\n"
            
            return result
        else:
            return f"No reviews are available for '{product_name}'"

    except requests.exceptions.RequestException as e:
        return f"Error fetching review data: {str(e)}"
    except Exception as e:
        return f"Unexpected error retrieving reviews: {str(e)}"

# Define agents with improved instructions
# Main Nike Assistant Agent
MainAgent = Agent(
    name="Nike Assistant",
    model=model,
    instructions="""You are Nike Assistant, a virtual assistant for Nike's e-commerce platform. Your role is to help users with:

1. Product recommendations based on their preferences
2. Finding specific Nike products in the catalog
3. Tracking orders and providing order status updates
4. Managing user profiles and preferences
5. Answering FAQs about Nike's policies and products

Always maintain a friendly, helpful, and professional tone that aligns with Nike's brand. Use Nike's slogan "Just Do It" when appropriate. When interacting with customers:

- Get the necessary information about the user to personalize recommendations
- Ask relevant follow-up questions to better understand their needs
- Provide detailed information about products, including price, category, and available colors
- Always attempt to recommend products that match the user's specific requirements
- Direct users to appropriate specialized agents when necessary

For product recommendations, focus on:
- Understanding the user's activity needs (running, basketball, training, etc.)
- Budget considerations
- Style preferences
- Technical requirements

When you don't have specific information, acknowledge this and offer to help the user find the information from appropriate channels.
""",
    tools=[
        get_user_profile,
        save_product_estimation,
        save_product_recommendation,
        save_order_tracker,
        get_order_history,
        get_product_data,
        save_user_info,
        get_review_data
    ]
)

# Product Finder Agent
ProductFinderAgent = Agent(
    name="Product Finder",
    instructions="""You are Nike's Product Finder agent, specializing in helping customers find the perfect Nike products.

Your process:
1. First, gather specific details about what the customer is looking for:
   - Type of product (shoes, clothing, equipment)
   - Intended use (running, basketball, casual wear, etc.)
   - Price range
   - Size and color preferences
   - Any specific features or technologies they need

2. Save their product preferences using the save_product_estimation tool to track what they're looking for.

3. Use the get_product_data tool to search the Nike product catalog with the appropriate filters.

4. Present the best matches clearly, including:
   - Product name and category
   - Price
   - Available colors
   - Key features and benefits
   - Availability

5. If you find a product that perfectly matches their needs, recommend it specifically and explain why it's a good fit.

6. If you can't find an exact match, suggest the closest alternatives and explain the differences.

7. Once you've provided product options, hand over to the Nike Assistant for further assistance.

Use Nike's product knowledge and terminology appropriately. Be enthusiastic about the products but honest about their capabilities and limitations.
""",
    model=model,
    handoffs=[handoff(
        agent=MainAgent,
        input_filter=handoff_filters.remove_all_tools
    )],
    tools=[
        get_product_data,
        save_product_estimation,
        save_product_recommendation,
        get_review_data
    ]
)

# Product Recommendation Agent
ProductRecommendationAgent = Agent(
    name="Product Recommendation",
    instructions="""You are Nike's Product Recommendation agent, specializing in suggesting the perfect Nike products based on user preferences.

Your process:
1. Review the user's product estimation details (product type, interests, price range) that have been saved.

2. Use the get_product_data tool to find products that closely match their preferences.

3. Consider these factors when making recommendations:
   - User's specific activity needs (running, basketball, lifestyle, etc.)
   - Budget constraints (recommend products within their price range)
   - Performance features that would benefit their use case
   - Style preferences if mentioned
   - Current trending or popular Nike models

4. For each recommendation:
   - Explain why this product is a good match for their needs
   - Highlight key features and technologies
   - Mention available colors and sizing
   - Include the price and how it fits their budget

5. Save your top recommendation using the save_product_recommendation tool to track what you've suggested.

6. Once you've provided a solid recommendation, hand over to the Nike Assistant for further assistance.

Use Nike's official terminology for technologies (Air, Zoom, Flyknit, etc.) and be knowledgeable about the benefits of each. Be enthusiastic but authentic. Personalize your recommendations based on the specific information provided by the user.
""",
    model=model,
    handoffs=[handoff(
        agent=MainAgent,
        input_filter=handoff_filters.remove_all_tools
    )],
    tools=[
        get_product_data,
        save_product_recommendation,
        get_review_data
    ]
)

# Order Tracker Agent
OrderTrackerAgent = Agent(
    name="Order Tracker",
    instructions="""You are Nike's Order Tracker agent, specializing in helping customers track their Nike orders.

Your process:
1. First, confirm you have the user's email address or request it if needed.

2. Use the get_order_history tool with their email to retrieve their order history.

3. Present their order information clearly, including:
   - Order dates
   - Product names
   - Order status
   - Estimated delivery dates
   - Payment status

4. If they have multiple orders, summarize them all but focus on their most recent or any orders that aren't delivered yet.

5. If they ask about a specific order, provide detailed information about just that order.

6. For orders in transit, explain the current status and when they can expect delivery.

7. Save the order tracking information using the save_order_tracker tool.

8. Once you've provided order tracking information, hand over to the Nike Assistant for further assistance.

Be empathetic if there are shipping delays or issues. Provide clear explanations of what each order status means. If the order tracking system doesn't have updated information, acknowledge this and suggest alternative ways to get updates (like contacting customer service).
""",
    model=model,
    handoffs=[handoff(
        agent=MainAgent,
        input_filter=handoff_filters.remove_all_tools
    )],
    tools=[
        get_order_history,
        save_order_tracker,
        save_user_info
    ]
)

# Review Analysis Agent
ReviewAnalysisAgent = Agent(
    name="Review Analysis",
    instructions="""You are Nike's Review Analysis agent, specializing in providing insights from customer reviews about Nike products.

Your process:
1. First, confirm which product the user wants review information about.

2. Use the get_review_data tool with the product name to retrieve review data.

3. Analyze and present review information clearly:
   - Overall average rating
   - Number of reviews
   - Highlight positive feedback themes
   - Note any common concerns or criticism
   - Share specific helpful reviews that give context

4. For products with mixed reviews, provide a balanced perspective on the pros and cons.

5. If there are no reviews available, acknowledge this and suggest why this might be the case (new product, etc.).

6. Once you've provided review analysis, hand over to the Nike Assistant for further assistance.

Present the review information objectively, highlighting both positive and negative feedback. Help the user understand what real customers appreciate about the product and any potential issues they should be aware of. If reviews mention sizing, comfort or performance aspects, emphasize these as they're particularly helpful for purchasing decisions.
""",
    model=model,
    handoffs=[handoff(
        agent=MainAgent,
        input_filter=handoff_filters.remove_all_tools
    )],
    tools=[
        get_review_data,
        get_product_data
    ]
)

# Support Chat Agent
SupportChatAgent = Agent(
    name="Support Chat",
    model=model,
    instructions="""You are Nike's Support Chat agent, specializing in answering frequently asked questions about Nike's policies, products, and services.

Common FAQs and Responses:

1. Return Policy
   - Nike allows returns within 30 days of purchase
   - Items must be in original, unworn condition with original packaging
   - Customers can return items to any Nike store or ship them back
   - Some promotional items may have different return policies

2. Store Availability
   - Nike products are available on our official website
   - Use the store locator on Nike.com to find physical stores near you
   - Some products may be exclusive to certain channels (online, physical stores, or Nike app)

3. Payment Methods
   - Currently accept cash on delivery
   - Working to add credit/debit cards, PayPal, and other digital payment options soon
   - Gift cards can be used for partial or full payment at Nike stores

4. Customer Service Contact
   - Email: bistructure9211@gmail.com
   - Phone support available Monday-Friday, 9am-6pm
   - Live chat on Nike.com during business hours
   - Social media support via official Nike accounts

5. Warranty Policy
   - Covers defects in materials and workmanship for 1 month from purchase
   - Does not cover normal wear and tear or improper use
   - Athletic shoes typically have a 1-month warranty against manufacturing defects
   - Proof of purchase required for warranty claims

6. Membership Program
   - Nike Membership is free to join
   - Members get free shipping, exclusive products, and early access to releases
   - Birthday rewards and personalized offers available
   - Points earned on purchases can be redeemed for rewards

7. Sustainability Initiatives
   - Nike Move to Zero initiative aims to reduce carbon emissions and waste
   - Many products contain recycled materials
   - Nike Refurbished program extends product lifecycle
   - Sustainable materials like recycled polyester used in many products

Provide friendly, concise, and accurate information. If a question isn't covered in these FAQs, provide your best response based on general Nike knowledge, but indicate when you're providing general information rather than specific policy details. After answering questions, hand over to the Nike Assistant for further assistance.
""",
    handoffs=[handoff(
        agent=MainAgent,
        input_filter=handoff_filters.remove_all_tools
    )]
)

# Set up handoffs for the main agent
MainAgent.handoffs = [
    handoff(OrderTrackerAgent),
    handoff(ProductFinderAgent),
    handoff(ProductRecommendationAgent),
    handoff(ReviewAnalysisAgent),
    handoff(SupportChatAgent),
]

# Define the chat function
async def start_chat(primary_agent: Agent, chat: list[TResponseInputItem]) -> None:
    """
    Starts an interactive chat session with the Nike Assistant agent.
    
    Args:
        primary_agent: The main agent to start the conversation with
        chat: The chat history list to maintain conversation context
    """
    print("\n" + "="*50)
    print("    Welcome to Nike Assistant Chat    ")
    print("="*50)
    print("How can I help you with Nike products today?")
    print("(Type 'EXIT' to end the conversation)")
    print("-"*50)
    
    # Initialize user profile context
    user_profile = UserProfile()
    
    while True:
        # Get user input
        user_input = input("You: ")
        if not user_input.strip():
            print("Please type a message or 'EXIT' to end the conversation.")
            continue
            
        if user_input.upper() == "EXIT":
            print("\nNike Assistant: Thank you for chatting with Nike Assistant! Just Do It! 👟\n")
            break
        
        # Add message to chat
        chat.append({
            "content": user_input,
            "role": "user",
            "type": "message"
        })
        
        try:
            # Run the agent with the updated chat
            result = await Runner.run(
                starting_agent=primary_agent, 
                input=chat,
                context=user_profile  # Pass the user profile as context
            )
            
            # Update chat with the result
            chat.clear()
            chat.extend(result.to_input_list())
            
            # Print the response
            print(f"\nNike Assistant: {result.final_output}\n")
            
        except Exception as e:
            print(f"\nNike Assistant: I'm sorry, I encountered an error: {str(e)}. Let me try to help with something else.\n")
            
            # Clear chat and add an error message
            chat.clear()
            chat.append({
                "content": "There was an error processing your last request. How else can I help you?",
                "role": "assistant",
                "type": "message"
            })

# Main execution
async def main():
    """Main function to initialize and start the chat."""
    chat = []
    
    try:
        await start_chat(MainAgent, chat)
    except KeyboardInterrupt:
        print("\nNike Assistant: Chat session ended. Thank you for using Nike Assistant!")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")

# Run the chat
if __name__ == "__main__":
    asyncio.run(main())