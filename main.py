from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from data import load_data
from langchain_core.prompts import load_prompt

# Load data and environment variables
load_dotenv()
data = load_data()

# Model from hf
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)
# Loading prompts
prompt1 = load_prompt('template_1.json')
prompt2 = load_prompt('template_2.json')


sample_review = data['flat_reviews'].iloc[0]


chain1 = prompt1 | model
result1 = chain1.invoke({"review_text": sample_review})

print(f"Review: '{sample_review}'")
print("-" * 25)
print(f"Result from Prompt 1 (Direct):")
print(result1.content)



chain2 = prompt2 | model
result2 = chain2.invoke({"review_text": sample_review})

print(f"\nResult from Prompt 2 (Role-Playing):")
print(result2.content)
