# 🎬 Sentiment Analysis of *The Exorcist* Series Reviews

**TL;DR:** An end-to-end NLP pipeline analyzing Rotten Tomatoes reviews of *The Exorcist*. Built with **LangChain + Hugging Face** to compare prompt strategies (direct vs. role-playing), showcasing how prompting affects model reliability.

---

## 🖼️ Project Workflow

![Flowchart](docs/flowchart.png)
*(see [`docs/NLP Analysis of _The Exorcist_ Series Reviews.pdf`](docs/NLP%20Analysis%20of%20_The%20Exorcist_%20Series%20Reviews.pdf) for full detail)*

---

## 📂 Repository Structure

```
.
├── preprocessing/                     # Notebooks for scraping & cleaning
├── src/                               # Core Python scripts
├── templates/                         # Prompt templates
├── data/                              # Final dataset
├── docs/                              # Visuals & reports
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 🕸️ Data Collection & Preprocessing

Notebooks inside [`preprocessing/`](preprocessing):

* **`web-scraping-movie-reviews.ipynb`** → Scrapes *The Exorcist* reviews from Rotten Tomatoes.
* **`sentiment-analysis-of-the-exorcist-reviews.ipynb`** → Cleans and preprocesses the raw text (lowercasing, stopwords, punctuation, tokenization, lemmatization).

Final output: `data/processed_data.csv`

---

## ⚙️ Workflow

1. **Scraping** → Collect reviews from Rotten Tomatoes.
2. **Preprocessing** → Clean & normalize text.
3. **Prompt Engineering** → Two strategies:

   * Direct prompt (concise).
   * Role-playing prompt (persona-based).
4. **Model Interaction** → Hugging Face LLM: `meta-llama/Llama-3.1-8B-Instruct`.
5. **Evaluation & Troubleshooting** → Compare outputs, document errors, assess hallucinations.

---

## 💻 Example Code

### Preprocessing (`src/data.py`)

```python
import pandas as pd
def load_data ():
    data = pd.read_csv("processed_data.csv")
    return data
```

### Sentiment Inference (`src/main.py`)

```python
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
```

---

## 📊 Example Output

**Sample Review:**

```
"brings,back,,original,horror,,73"
```

* **Direct Prompt** ✅

  ```
  Sentiment: POSITIVE
  Reason: Nostalgic reference to “original horror” implies enthusiasm.
  ```

* **Role-Playing Prompt** ❌

  ```
  Sentiment: NEGATIVE
  Reason: Over-interpreted punctuation → hallucinated disappointment.
  ```

👉 Demonstrates how **prompt design directly impacts LLM reliability**.

---

## 🚀 How to Run

```bash
git clone https://github.com/lakshayknows/sentiment-analysis-the_exorcist_series.git
cd sentiment-analysis-the_exorcist_series
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Add Hugging Face API token in `.env`:

```ini
HUGGINGFACEHUB_API_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"
```

Run:

```bash
python src/app.py
```

---

## 🛠️ Tools & Libraries

* **Python**
* **pandas, NLTK, TextBlob** → preprocessing
* **LangChain** → prompt orchestration
* **Hugging Face Hub** → meta-llama LLM
* **Napkin.ai** → flowchart design

---

## 🎯 Key Takeaways

* Direct prompts → **stable results**
* Persona prompts → **risk of hallucinations**
* Preprocessing + prompt strategy → **better reliability**

---

## 📜 License

Licensed under the [MIT License](LICENSE).

---

✨ *From raw web-scraped chaos to model-guided clarity — an exorcism of noisy data into sentiment truth.* 👻

---

Would you like me to also **generate the new folder structure + move commands** (bash `mkdir`, `mv`) so you can reorganize your repo quickly without doing it manually in GitHub?
