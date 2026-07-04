from dotenv import load_dotenv
load_dotenv(override=True)

from langchain.tools import tool

@tool
def get_length(text: str) -> int:
    """Determine the length of a string."""
    return len(text)

from rich import print

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-5-mini")

llm_pwrd_tool = llm.bind_tools([get_length])

# response = llm_pwrd_tool.invoke("hello there, whats the length if this prompt?")

# if response.tool_calls:
#     tool_call = response.tool_calls[0]
#     tool_result = get_length.invoke(tool_call['args'])

#     final_response = llm.invoke(
#         f"the length of the text is {tool_result}"
#     )
#     # print(final_response.content) 

# print(get_length.invoke(response.tool_calls[0]))

tools = {
    "get_length" : get_length
}

from langchain.messages import HumanMessage

message = []
query = HumanMessage(content="return number of characters in the given text along with a joke based on the length: 'hello how are you?'")
message.append(query)

ai_response = llm_pwrd_tool.invoke(message)

message.append(ai_response)

if ai_response.tool_calls:
    tool_name = ai_response.tool_calls[0]["name"]
    tool_message = tools[tool_name].invoke(ai_response.tool_calls[0])
    message.append(tool_message)

response = llm_pwrd_tool.invoke(message)
print(response.content)
