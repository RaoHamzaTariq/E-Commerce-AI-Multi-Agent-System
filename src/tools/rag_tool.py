from rag.retriever import retriever
from agents import function_tool

@function_tool
async def rag_tool(prompt:str)->str:
    """
    Retrieves a response from the chromadb based on the input prompt.
    Args:
    prompt (str): The input prompt to be used for retrieval.
    Returns:
    str: The retrieved response from the chromadb.
    """
    try:
        result = await retriever(prompt)
        return result
    except Exception as e:
        return str(e)