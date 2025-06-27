from .loader import collection
import asyncio

async def retriever(prompt :str)->str:
    
    results = collection.query(
        query_texts=[prompt],
        n_results=1
    )
    return results["documents"][0][0]

# print(asyncio.run(retriever(prompt="How can customers contact for support?")))