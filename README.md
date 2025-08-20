🎬 Sentiment Analysis of The Exorcist Series Reviews

TL;DR: An end-to-end NLP pipeline analyzing Rotten Tomatoes reviews of The Exorcist. Built with LangChain + Hugging Face to compare prompt strategies (direct, role-playing, and few-shot), showcasing how prompting affects model reliability.

🖼️ Project Workflow


(see docs/NLP Analysis of _The Exorcist_ Series Reviews.pdf
 for full detail)

📂 Repository Structure
.
├── preprocessing/                     # Notebooks for scraping & cleaning
│   ├── web-scraping-movie-reviews.ipynb
│   └── sentiment-analysis-of-the-exorcist-reviews.ipynb
│
├── src/                               # Core Python scripts
│   ├── app.py
│   ├── main.py
│   ├── data.py
│   └── create_template.py
│
├── templates/                         # Prompt templates
│   ├── template_1.json   # Direct
│   ├── template_2.json   # Role-playing
│   └── template_3.json   # Few-shot
│
├── data/                              # Final dataset
│   └── processed_data.csv
│
├── docs/                              # Visuals & reports
│   ├── NLP Analysis of _The Exorcist_ Series Reviews.pdf
│   └── flowchart.png
│
├── requirements.txt
├── README.md
└── LICENSE

🕸️ Data Collection & Preprocessing

Notebooks inside preprocessing/
:

web-scraping-movie-reviews.ipynb → Scrapes The Exorcist reviews from Rotten Tomatoes.

sentiment-analysis-of-the-exorcist-reviews.ipynb → Cleans and preprocesses the raw text (lowercasing, stopwords, punctuation, tokenization, lemmatization).

Final output: data/processed_data.csv

⚙️ Workflow

Scraping → Collect reviews from Rotten Tomatoes.

Preprocessing → Clean & normalize text.

Prompt Engineering → Three strategies:

Direct prompt (concise).

Role-playing prompt (persona-based).

Few-shot prompt (anchored with labeled examples).

Model Interaction → Hugging Face LLM: meta-llama/Llama-3.1-8B-Instruct.

Evaluation & Troubleshooting → Compare outputs, document errors, assess hallucinations.

💻 Example Code
Preprocessing (src/data.py)
import pandas as pd
def load_data():
    data = pd.read_csv("data/processed_data.csv")
    return data

Sentiment Inference (src/main.py)
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from data import load_data
from langchain_core.prompts import load_prompt

# Load data and environment variables
load_dotenv()
data = load_data()

# Model from Hugging Face
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

# Load prompt templates
prompt1 = load_prompt('templates/template_1.json')
prompt2 = load_prompt('templates/template_2.json')
prompt3 = load_prompt('templates/template_3.json')

sample_review = data['flat_reviews'].iloc[0]

print(f"Review: '{sample_review}'")
print("-" * 40)

# Direct Prompt
chain1 = prompt1 | model
result1 = chain1.invoke({"review_text": sample_review})
print("\nResult from Prompt 1 (Direct):")
print(result1.content)

# Role-Playing Prompt
chain2 = prompt2 | model
result2 = chain2.invoke({"review_text": sample_review})
print("\nResult from Prompt 2 (Role-Playing):")
print(result2.content)

# Few-Shot Prompt
chain3 = prompt3 | model
result3 = chain3.invoke({"review_text": sample_review})
print("\nResult from Prompt 3 (Few-Shot):")
print(result3.content)

📊 Example Output

Sample Review:

"brings,back,,original,horror,,73"


Direct Prompt ✅

Sentiment: POSITIVE
Reason: Nostalgic reference to “original horror” implies enthusiasm.


Role-Playing Prompt ❌

Sentiment: NEGATIVE
Reason: Over-interpreted punctuation → hallucinated disappointment.


Few-Shot Prompt ✅

Sentiment: POSITIVE
Reason: Anchored by examples, the model avoids over-interpretation and classifies correctly.


👉 Demonstrates how prompt design directly impacts LLM reliability.

🚀 How to Run
git clone https://github.com/lakshayknows/sentiment-analysis-the_exorcist_series.git
cd sentiment-analysis-the_exorcist_series
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt


Add Hugging Face API token in .env:

HUGGINGFACEHUB_API_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"


Run:

python src/main.py

🛠️ Tools & Libraries

Python

pandas, NLTK, TextBlob → preprocessing

LangChain → prompt orchestration

Hugging Face Hub → meta-llama LLM

Napkin.ai → flowchart design

🎯 Key Takeaways

Direct prompts → stable results

Persona prompts → risk of hallucinations

Few-shot prompts → reduce hallucinations, improve reliability

Preprocessing + prompt strategy → robust sentiment classification

📜 License

Licensed under the MIT License
.

✨ From raw web-scraped chaos to model-guided clarity — an exorcism of noisy data into sentiment truth. 👻
