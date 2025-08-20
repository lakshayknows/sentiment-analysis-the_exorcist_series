from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from data import load_data
from langchain_core.prompts import load_prompt

# Load data and environment variables
load_dotenv()
data = load_data()

# Hugging Face model
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

# Load prompt templates
prompt1 = load_prompt('template_1.json')
prompt2 = load_prompt('template_2.json')
prompt3 = load_prompt('template_3.json')


sample_review = data['flat_reviews'].iloc[0]

print(f"Review: '{sample_review}'")
print("-" * 40)


chain1 = prompt1 | model
result1 = chain1.invoke({"review_text": sample_review})
print("\nResult from Prompt 1 (Direct):")
print(result1.content)


chain2 = prompt2 | model
result2 = chain2.invoke({"review_text": sample_review})
print("\nResult from Prompt 2 (Role-Playing):")
print(result2.content)


chain3 = prompt3 | model
result3 = chain3.invoke({"review_text": sample_review})
print("\nResult from Prompt 3 (Few-Shot):")
print(result3.content)
