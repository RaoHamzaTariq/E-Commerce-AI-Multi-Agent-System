import requests
import pandas as pd
from agents import function_tool

@function_tool
def get_product_data()->str:
    """
    Fetches real-time product data from the website and writes it to a neatly formatted text file.

    Returns:
        str: product data of nike website 
    """
    try:
        url = "https://nike-marketplace-bi-structure.vercel.app/api/products"
        response = requests.get(url)
        response.raise_for_status()

        data = response.json()
        products = []

        for item in data.get("data", []):
            products.append({
                "Product Name": item.get("productName", ""),
                "Price": item.get("price", ""),
                "Inventory": item.get("inventory", ""),
                "Colors": ", ".join(item.get("colors", [])),
                "Status": item.get("status", ""),
                "Category": item.get("category", ""),
                # "Slug": item.get("slug", {}).get("current", "")
            })

        df = pd.DataFrame(products)

        # Write as clean, tab-separated text (no index, no padding)
        # df.to_csv("product_data.txt", sep="\t", index=False)

        # print(df.to_string(index=False))  # Optional: show clean table in terminal

        return df.to_string(index=False)

    except Exception as e:
        return f"Error: {e}"

# Run the function
if __name__ == "__main__":
    message = get_product_data()
    print(message)
