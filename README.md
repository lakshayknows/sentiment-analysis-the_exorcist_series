# 🎬 Sentiment Analysis of *The Exorcist* Series Reviews

**TL;DR:** An end-to-end NLP pipeline analyzing Rotten Tomatoes reviews of *The Exorcist*. Built with **LangChain + Hugging Face** to compare prompt strategies (direct, role-playing, and few-shot), showcasing how prompting affects model reliability.

---

## 🖼️ Project Workflow

![Flowchart](docs/flowchart.png)
*(see full PDF: [`docs/Flowchart_NLP_The_Exorcist_Series.pdf`](docs/Flowchart_NLP_The_Exorcist_Series.pdf))*

---

## 📂 Repository Structure

```
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
│   ├── Flowchart_NLP_The_Exorcist_Series.pdf
│   ├── flowchart.png
│   └── Model Evaluation and TroubleShooting.pdf
│
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
3. **Prompt Engineering** → Three strategies:

   * Direct prompt (concise).
   * Role-playing prompt (persona-based).
   * Few-shot prompt (anchored with labeled examples).
4. **Model Interaction** → Hugging Face LLM: `meta-llama/Llama-3.1-8B-Instruct`.
5. **Evaluation & Troubleshooting** → Compare outputs, document errors, assess hallucinations.

---

## 💻 Example Code

### Preprocessing (`src/data.py`)

```python
import pandas as pd
def load_data():
    data = pd.read_csv("data/processed_data.csv")
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
```

---

## 📊 Example Output

**Sample Review:**

```
"brings,back,,original,horror,,73"
```

### Result from Prompt 1 (Direct) ✅

```
The sentiment of this review is overwhelmingly positive. The reviewer is expressing nostalgia and appreciation for the return of the original horror elements, and the fact that it brings back memories of the past suggests a sense of satisfaction and delight. The punctuation and phrasing ("brings, back,,") also give the impression of excitement and enthusiasm.
```

### Result from Prompt 2 (Role-Playing) ❌

```
This review is extremely short and doesn't provide any clear opinion or analysis of the horror movie. It appears to be a random collection of words, possibly a play on the phrase "brings back the original horror" with a numerical value ("73") appended to the end.

As a horror fan, I'd say this review is more confusing than helpful. I wouldn't be able to determine if the reviewer liked or disliked the movie based on this review. It's unclear what the "73" refers to, and the use of multiple commas and lack of proper capitalization make it difficult to read and understand.

A good review should provide some insight into the movie's strengths and weaknesses, or at least convey the reviewer's opinion in a clear and concise manner. This review falls short of that standard.
```

### Result from Prompt 3 (Few-Shot) ✅

```
It seems like the review is referring to a movie, likely "Brings Back Original Horror" (assuming "73" is a rating out of 100). However, the text is incomplete and appears to be truncated.

That being said, I'll attempt to classify the sentiment based on the available information. Since the review seems to be making a positive statement about the movie's return to original horror, I would classify the sentiment as:

POSITIVE
```

---

## 📊 Quick Comparison Table

| Prompt Type  | Sentiment  | Notes                                                               |
| ------------ | ---------- | ------------------------------------------------------------------- |
| Direct       | ✅ Positive | Correct, stable, interprets enthusiasm.                             |
| Role-Playing | ❌ Negative | Over-analyzes punctuation, hallucinates confusion.                  |
| Few-Shot     | ✅ Positive | Anchored by examples, correct classification but misreads metadata. |

👉 This shows both the **actual model responses** and your **evaluation at a glance**.

---

## 📑 Evaluation & Troubleshooting

Detailed notes are available in [`docs/Model Evaluation and TroubleShooting.pdf`](docs/Model%20Evaluation%20and%20TroubleShooting.pdf).

* **Evaluation:** Compared Direct, Role-Playing, and Few-Shot prompts qualitatively.
* **Findings:**

  * Direct prompt = most reliable.
  * Role-playing prompt = over-analyzed punctuation, hallucinated errors.
  * Few-shot prompt = more grounded, but still misinterpreted some metadata.
* **Troubleshooting:**

  * Issue: Small dataset (220 reviews) → risk of bias & underfitting.
  * Solution: Use synthetic data generation with **Faker** to augment training data.

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
python src/main.py
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
* Few-shot prompts → **reduce hallucinations, improve reliability**
* Small datasets can introduce **bias & underfitting** → solved via augmentation

---

## 📜 License

Licensed under the [MIT License](LICENSE).

---

✨ *From raw web-scraped chaos to model-guided clarity — an exorcism of noisy data into sentiment truth.* 👻

