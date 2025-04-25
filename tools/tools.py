import json
from typing import List
from agent import Agent, FunctionTool, RunContextWrapper, function_tool, Runner, TResponseInputItem
import pandas as pd
import requests

from data.dataModel import OrderDetail, OrderTracker, ProductEstimation, ProductRecommendation, UserInfo, UserProfile

@function_tool
async def get_user_profile(wrapper:RunContextWrapper[UserProfile]):
    """
    Get user profile information.
    """
    try:
        return("User Profile: ", wrapper.context.model_dump_json())
    except Exception as e:
        print(f"Failed to view user profile: {e}")
        raise

@function_tool
async def save_product_estimation(wrapper:RunContextWrapper[UserProfile],product_name:str, product_type:str, price_range:int):
    """
    Save product estimation information of user.
    Args:
        product_name (str): Name of the product.
        product_type (str): Type of the product.
        price_range (int): Price range of the product.
    Returns:
        str: Confirmation message.
    Raises:
        Exception: If there was an error saving the product estimation plan
    """
    try:
        wrapper.context.product_estimation = ProductEstimation(
            product_name=product_name,
            product_type=product_type,
            product_price=price_range
        )
        return("Product Estimation Saved: ", wrapper.context.product_estimation.model_dump_json())
    except Exception as e:
        print(f"Failed to save product estimation: {e}")
        raise

@function_tool
async def save_product_recommendation(wrapper:RunContextWrapper[UserProfile],product_name:str, product_type:str, product_price:int, product_link:str):
    """
    Save product recommendation information of user.
    Args:
        product_name (str): Name of the product.
        product_type (str): Type of the product.
        product_price (int): Price of the product.
        product_link (str): Link to the product.
    Returns:
        str: Confirmation message.
    Raises:
        Exception: If there was an error saving the product recommendation
    """
    try:
        wrapper.context.product_recommendation = ProductRecommendation(
            product_name=product_name,
            product_type=product_type,
            product_price=product_price,
            product_link=product_link
        )
        return("Product Recommendation Saved: ", wrapper.context.product_recommendation.model_dump_json())
    except Exception as e:
        print(f"Failed to save product recommendation: {e}")
        raise

@function_tool
async def save_order_tracker(wrapper:RunContextWrapper[UserProfile],total_orders:int, orders_detail:List[OrderDetail]):
    """
    Save order tracker information of user.
    Args:
        total_orders (int): Total number of orders.
        orders_detail (List[OrderDetail]): List of order details.
    Returns:
        str: Confirmation message.
    Raises:
        Exception: If there was an error saving the order tracker
    """
    try:
        wrapper.context.order_tracker = OrderTracker(
            total_orders=total_orders,
            orders_detail=orders_detail
        )
        return("Order Tracker Saved: ", wrapper.context.order_tracker.model_dump_json())
    except Exception as e:
        print(f"Failed to save order tracker: {e}")
        raise

@function_tool
async def get_order_history(wrapper:RunContextWrapper[UserProfile], user_email: str) -> str:
    """
    Retrieves order history data for a given user email from an API and returns it as a Pandas DataFrame string.

    Args:
        user_email: The email address of the user.

    Returns:
        A string representation of a Pandas DataFrame containing order details
        (id, order_date, product_name, price, quantity, sub_total, paymentStatus, orderStatus).
        Returns an error message as a string if there are issues with the API request or data processing.
    """
    try:
        response = await requests.get(f"http://localhost:3000/api/users?email={user_email}")
        response.raise_for_status()  # Raise an exception for bad status codes

        data = await response.json()['data']
        data_frame = {
            "order_date": [],
            "product_name": [],
            "price": [],
            "quantity": [],
            "sub_total": [],
            "paymentStatus": [],
            "orderStatus": []
        }
        for order in data.get('orderHistory', []):  # Use .get() to handle missing key
            for item in order.get('productDetails', []):  # Use .get() to handle missing key
                data_frame['order_date'].append(pd.to_datetime(order.get('orderDate')).date())
                data_frame['product_name'].append(item['product_id'].get('productName'))
                data_frame['price'].append(item['product_id'].get('price'))
                data_frame['quantity'].append(item.get('quantity'))
                data_frame['sub_total'].append(item.get('subtotal'))
                data_frame['paymentStatus'].append(order.get('paymentStatus'))
                data_frame['orderStatus'].append(order.get('orderStatus'))

        df = pd.DataFrame(data_frame)
        return str(df)

    except requests.exceptions.RequestException as e:
        return f"Error: API request failed - {e}"
    except (KeyError, ValueError) as e:
        return f"Error: Could not process API response - {e}"
    except Exception as e:
        return f"Error: An unexpected error occurred - {e}"

@function_tool()
async def get_product_data(wrapper:RunContextWrapper[UserProfile]):
    """
    Fetches product data from an API and returns it as a Pandas DataFrame.
    """
    response = await requests.get("https://nike-marketplace-bi-structure.vercel.app/api/products")
    data = await response.json()
    products = []
    for item in data["data"]:
        products.append({
            "Product Name": item.get("productName", ""),
            "Price": item.get("price", ""),
            "Inventory": item.get("inventory", ""),
            "Colors": ", ".join(item.get("colors", [])),  # Convert list to comma-separated string
            "Status": item.get("status", ""),
            "Category": item.get("category", ""),
            "Slug": item.get("slug", {}).get("current", "")
        })

    # Convert to DataFrame
    df = pd.DataFrame(products)
    return df

@function_tool
async def save_user_info(wrapper:RunContextWrapper[UserProfile], name:str, age:int, email:str):
    """
    Save user information.

    Args:
        name (str): Name of the user.
        age (int): Age of the user.
        email (str): Email of the user.
    Returns:
        str: Confirmation message.
    """
    try:
        wrapper.context.user_info = UserInfo(
            name=name,
            age=age,
            email=email
        )
        return("User Information Saved: ", wrapper.context.user_info.model_dump_json())
    except Exception as e:
        print(f"Failed to save user information: {e}")
        raise

@function_tool
async def get_review_data(wrapper:RunContextWrapper[UserProfile],product_name: str) -> str:
    """
    Retrieves review data (comment and rating) for a given product name from an API.

    Args:
        product_name: The name of the product to fetch reviews for.

    Returns:
        A string representation of a Pandas DataFrame containing 'comment' and 'rating'
        if reviews are found. Returns "No review and rating available" if no reviews
        are found, or "Error: Product not found" if the API request fails.
    """
    try:
        list_of_words = product_name.split(" ")
        lower_list = [part.lower() for part in list_of_words]
        slug = "-".join(lower_list)
        slug = slug.replace("'", "")

        response = requests.get(f"https://nike-marketplace-bi-structure.vercel.app/api/products?slug={slug}")
        response.raise_for_status()  # Raise an exception for bad status codes

        data = response.json()
        if "data" in data and "reviews" in data["data"] and data["data"]["reviews"]:
            reviews_data = data["data"]["reviews"]
            df_reviews = pd.DataFrame(reviews_data)
            df_review_rating = df_reviews[["comment", "rating"]]
            return str(df_review_rating)
        else:
            return "No review and rating available"

    except requests.exceptions.RequestException as e:
        return f"Error: API request failed - {e}"
    except (KeyError, ValueError):
        return "Error: Invalid API response format"