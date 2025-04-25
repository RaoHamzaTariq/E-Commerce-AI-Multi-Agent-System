from dataclasses import dataclass
from typing import List

@dataclass
class UserInfo:
    name: str
    age: int
    email: str

@dataclass
class ProductEstimation:
    product_name: str
    product_type: str
    price_range: int

@dataclass
class ProductRecommendation:
    product_name: str
    product_type: str
    product_price: int
    product_slug: str

@dataclass
class OrderDetail:
    order_status: str
    estimated_delivery_date: str
    no_of_products: int
    total_price: int

@dataclass
class OrderTracker:
    total_orders: int
    orders_detail: List[OrderDetail]

@dataclass
class UserProfile:
    user_info: UserInfo
    product_estimation: ProductEstimation
    product_recommendation: ProductRecommendation
    order_tracker: OrderTracker
