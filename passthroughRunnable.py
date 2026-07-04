from dotenv import load_dotenv
load_dotenv(override=True)

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-5-mini")

from langchain_core.prompts import ChatPromptTemplate

gen_code = ChatPromptTemplate.from_messages([
    ("system", "You are a code generator, which generates only the compilable code with no comments, nothing else"),
    ("human", "{topic}")
])

explain_code = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant who explaiins code in simple terms"),
    ("human", "Explain the code in Simple words:\n{code}")
])


from langchain_core.output_parsers import StrOutputParser
parser = StrOutputParser()

from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough


chain = gen_code | llm | parser 
chain2 = RunnableParallel({
    "code": RunnablePassthrough(),
    "explanation": explain_code | llm | parser
})

seq = chain | chain2
# | (lambda code: {"code": code}) | explain_code | llm | parser

result = seq.invoke({"topic": "write a code of three sum in java"})

print(f"code -> {result['code']}\n\n")
print(f"explanation -> {result['explanation']}")