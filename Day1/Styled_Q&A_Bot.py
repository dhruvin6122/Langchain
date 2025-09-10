from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate



# Initialize Ollama model
llm = OllamaLLM(model="llama3.2:3b-instruct-q4_0")

# Define prompt template
template = """
You are a helpful assistant. Answer the question in this style: {style}.
Question: {question}
"""
prompt = PromptTemplate(input_variables=["style", "question"], template=template)

# Create a runnable sequence
chatbot = prompt | llm

print("Welcome to Chat in Style! Type 'exit' to quit.")

while True:
    question = input("Enter your question: ")
    if question.lower() == "exit":
        print("Goodbye!")
        break
    style = input("Enter style (funny/formal/kid/simple/exam): ")
    response = chatbot.invoke({"style": style, "question": question})
    print("Bot:", response)
