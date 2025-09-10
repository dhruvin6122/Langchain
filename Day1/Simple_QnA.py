from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

# Initialize Ollama model
llm = OllamaLLM(model="llama3.2:3b-instruct-q4_0")

# Prompt template
prompt = PromptTemplate(
    input_variables=["question"],
    template="Answer this question: {question}"
)

# Create a chain
chain = prompt | llm

# Run
answer = chain.invoke("Who is Mahatma Gandhi?")
print(answer)
