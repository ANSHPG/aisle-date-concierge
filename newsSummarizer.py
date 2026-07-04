from dotenv import load_dotenv
load_dotenv(override=True)

from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

tool = TavilySearch(
    max_results=4,
    topic="news"
)

llm = ChatOpenAI(model = "gpt-5-mini") 

prompt = ChatPromptTemplate.from_template(
    """
    You are a stock analyst
    You will be given news title and content regarding stock market in last 24 hours
    You will suggest what stocks among the news to invest or not and why oin bullet points

    News: {news}
"""
)

chain = prompt | llm | StrOutputParser()

result = tool.invoke({"query": "stock market news in last 24 hours"})
res_join = "\n"
for item in result["results"]:
    res_join += f"title:{item['title']}\ncontent:{item['content']}\n-------\n"

output = chain.invoke({"news": res_join})

print("News Summary:\n", output)