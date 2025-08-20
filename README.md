<img width="1148" height="691" alt="image" src="https://github.com/user-attachments/assets/c657a17c-5e41-4948-a551-ca4abba3a44b" /><img width="1148" height="691" alt="image" src="https://github.com/user-attachments/assets/0297cf11-2d36-4fa8-9996-b1e88a9625b2" />🎬 Sentiment Analysis of The Exorcist Series Reviews

TL;DR: An end-to-end NLP pipeline analyzing Rotten Tomatoes reviews of The Exorcist. Built with LangChain + Hugging Face to compare prompt strategies (direct vs. role-playing), showcasing how prompting affects model reliability.

🖼️ Project Workflow


(see docs/NLP Analysis of _The Exorcist_ Series Reviews.pdf
 for full detail)

📂 Repository Structure
.
├── preprocessing/                     # Notebooks for scraping & cleaning
├── src/                               # Core Python scripts
├── templates/                         # Prompt templates
├── data/                              # Final dataset
├── docs/                              # Visuals & reports
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

Prompt Engineering → Two strategies:

Direct prompt (concise).

Role-playing prompt (persona-based).

Model Interaction → Hugging Face LLM: meta-llama/Llama-3.1-8B-Instruct.

Evaluation & Troubleshooting → Compare outputs, document errors, assess hallucinations.

💻 Example Code
Preprocessing (src/data.py)
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

def clean_text(text):
    text = re.sub(r"http\S+|www\S+", "", text)   # remove URLs
    tokens = word_tokenize(text.lower())         # lowercase + tokenize
    lemmatizer = WordNetLemmatizer()
    cleaned = [lemmatizer.lemmatize(t) for t in tokens if t.isalpha()]
    return " ".join(cleaned)

Sentiment Inference (src/app.py)
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import json

# Load prompt
with open("templates/template_1.json") as f:
    prompt_dict = json.load(f)
prompt = PromptTemplate(**prompt_dict)

# Run LLM chain
chain = LLMChain(llm=hf_llm, prompt=prompt)
review_text = "brings,back,,original,horror,,73"
response = chain.run({"input": review_text})

print("Sentiment result:", response)

📊 Example Output

Sample Review:

"brings,back,,original,horror,,73"


Direct Prompt ✅

Sentiment: POSITIVE
Reason: Nostalgic reference to “original horror” implies enthusiasm.


Role-Playing Prompt ❌

Sentiment: NEGATIVE
Reason: Over-interpreted punctuation → hallucinated disappointment.


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

python src/app.py

🛠️ Tools & Libraries

Python

pandas, NLTK, TextBlob → preprocessing

LangChain → prompt orchestration

Hugging Face Hub → meta-llama LLM

Napkin.ai → flowchart design

🎯 Key Takeaways

Direct prompts → stable results

Persona prompts → risk of hallucinations

Preprocessing + prompt strategy → better reliability

📜 License

Licensed under the MIT License
.

✨ From raw web-scraped chaos to model-guided clarity — an exorcism of noisy data into sentiment truth. 👻
