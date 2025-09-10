from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel

# -----------------------------
# 1️⃣ Initialize Ollama (local model)
# -----------------------------
llm = OllamaLLM(model="llama3.2:3b-instruct-q4_0")  

# -----------------------------
# 2️⃣ Prompt templates
# -----------------------------
resume_prompt = PromptTemplate(
    input_variables=["name", "experience", "skills", "role"],
    template="""
Create professional resume bullet points for the following candidate:

Name: {name}
Experience: {experience}
Skills: {skills}
Role applying for: {role}
"""
)

cover_letter_prompt = PromptTemplate(
    input_variables=["name", "role", "resume"],
    template="""
Write a professional cover letter for {name} applying for the role of {role}.
Use the following resume bullet points as reference:
{resume}
"""
)

linkedin_prompt = PromptTemplate(
    input_variables=["name", "role", "resume"],
    template="""
Write a LinkedIn summary for {name} who is applying for {role}.
Use the following resume bullet points as guidance:
{resume}
"""
)

career_suggestion_prompt = PromptTemplate(
    input_variables=["skills", "role"],
    template="""
The candidate has skills: {skills}.
They want to apply for role: {role}.

Give clear suggestions on:
1. Which **next technical skills** they should learn to be industry ready.
2. Which **soft skills** (communication, teamwork, etc.) will help them.
3. How they can build a strong **portfolio** to get hired.
Keep it practical and short.
"""
)

# -----------------------------
# 3️⃣ User Input
# -----------------------------
name = input("Candidate Name: ")
experience = input("Experience (short description): ")
skills = input("Skills (comma separated): ")
role = input("Role applying for: ")

candidate_info = {
    "name": name,
    "experience": experience,
    "skills": skills,
    "role": role
}

# -----------------------------
# 4️⃣ Step1: Resume
# -----------------------------
resume_step = RunnableLambda(
    lambda x: llm.invoke(resume_prompt.format(
        name=x["name"],
        experience=x["experience"],
        skills=x["skills"],
        role=x["role"]
    ))
)

resume_text = resume_step.invoke(candidate_info)

print("\n✅ Resume:\n")
print(resume_text)

# -----------------------------
# 5️⃣ Step2, 3 & 4: Parallel (Cover Letter + LinkedIn + Career Suggestions)
# -----------------------------
parallel_steps = RunnableParallel({
    "cover_letter": RunnableLambda(
        lambda x: llm.invoke(cover_letter_prompt.format(
            name=x["name"],
            role=x["role"],
            resume=resume_text
        ))
    ),
    "linkedin_summary": RunnableLambda(
        lambda x: llm.invoke(linkedin_prompt.format(
            name=x["name"],
            role=x["role"],
            resume=resume_text
        ))
    ),
    "career_suggestions": RunnableLambda(
        lambda x: llm.invoke(career_suggestion_prompt.format(
            skills=x["skills"],
            role=x["role"]
        ))
    )
})

results = parallel_steps.invoke(candidate_info)

# -----------------------------
# 6️⃣ Final Output
# -----------------------------
print("\n✅ Cover Letter:\n")
print(results["cover_letter"])

print("\n✅ LinkedIn Summary:\n")
print(results["linkedin_summary"])

print("\n✅ Career Suggestions for You:\n")
print(results["career_suggestions"])
