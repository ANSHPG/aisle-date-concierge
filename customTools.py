from langchain.tools import tool

@tool 
def get_greeting(name) -> str:
    """Generate a greeting message for the user""" #docstring -> description
    return f"Hello, {name}! Welcome to Ai Cosmos"

res = get_greeting.invoke({"name": "Anshu"})
print(res)

print(get_greeting.name)
print(get_greeting.description)
print(get_greeting.args)