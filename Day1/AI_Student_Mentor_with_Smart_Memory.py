from langchain_ollama import OllamaLLM
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers.string import StrOutputParser





# 1️⃣ Initialize Local LLM (Ollama)
llm = OllamaLLM(model="llama3.2:3b-instruct-q4_0")

# 2️⃣ Hybrid Memory (Summary + Recent Window)
memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=250,   # limit for summary compression
    return_messages=True
)

# 3️⃣ Custom Prompt (Mentor Style)
mentor_prompt = PromptTemplate(
    input_variables=["history", "input"],
    template="""
You are an AI Student Mentor Named Careerbanao.ai created by Dhruvin patel AI badhshaa.
Your responsibilities:
- Understand student’s current skills and career goals
- Suggest next skills to learn (industry-relevant)
- Explain why these skills matter
- Keep answers short, practical, and motivating

Conversation so far:
{history}

Student: {input}
Mentor:"""
)
# Initialize parser
parser = StrOutputParser()

# 4️⃣ Build Conversation Chain
mentor_chain = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=mentor_prompt,
   # verbose=True   # shows internal prompt + memory (for learning/debugging)
)

# 5️⃣ Chat Loop
print("🎓 AI Student Mentor started (type 'exit' to quit)")
while True:
    user_input = input("Student: ")
    if user_input.lower() == "exit":
        print("👋 Goodbye! Keep practicing and learning consistently.")
        break
    response = mentor_chain.invoke(user_input)

    clean_text = parser.parse(response)
    print("Mentor:", clean_text['response'])
 

