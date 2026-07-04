from dotenv import load_dotenv
load_dotenv(override=True)

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-5-mini")

from langchain_core.prompts import PromptTemplate

short_prompt = PromptTemplate.from_template(
    "Explain {topic} in short and simple to a child"
)
detailed_prompt = PromptTemplate.from_template(
    "Explain {topic} in short to an adult in a mature language"
)

from langchain_core.output_parsers import StrOutputParser
parser = StrOutputParser()

from langchain_core.runnables import RunnableParallel, RunnableLambda

chain = RunnableParallel({
    "short": RunnableLambda(lambda x:x['short']) |short_prompt | llm | parser,
    "detailed": RunnableLambda(lambda x:x['detailed'])| detailed_prompt | llm | parser
})

result = chain.invoke({
    "short": {"topic": "Machine Learning"},
    "detailed": {"topic": "Deep Learning"}
})

print(f"short -> {result['short']}")
print(f"detailed -> {result['detailed']}")