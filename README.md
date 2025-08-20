Sentiment Analysis of "The Exorcist" Series ReviewsThis project is a submission for an NLP internship assignment. It demonstrates a complete, end-to-end workflow for a sentiment analysis task, including data scraping, extensive preprocessing, and prompt engineering with LangChain to interact with a large language model from the Hugging Face Hub.The primary submission is a visual flowchart that details the project's lifecycle, accompanied by this repository containing all the source code and documentation.📂 Repository Structure.
├── .gitignore
├── LICENSE
├── README.md
├── app.py
├── create_templates.py
├── data.py
├── processed_data.csv
├── requirements.txt
├── template_1.json
├── template_2.json
├── flowchart.pdf
└── evaluation_and_troubleshooting.md
app.py: The main application script that loads the processed data and prompt templates, interacts with the Hugging Face model, and prints the final sentiment analysis results.data.py: A utility script responsible for loading the raw review data and applying all preprocessing steps.create_templates.py: A helper script to generate and save the LangChain PromptTemplate objects into reusable JSON files.processed_data.csv: The final, cleaned, and lemmatized dataset that is used as the input for the main application.template_1.json & template_2.json: The saved prompt templates that define the different interaction strategies with the language model.requirements.txt: A list of all the Python libraries required to run this project.flowchart.pdf: The main visual submission detailing the project workflow from start to finish.evaluation_and_troubleshooting.md: A document containing the detailed written analysis for steps 3 (Evaluation) and 4 (Troubleshooting) of the assignment.📊 Final Output AnalysisThe script was run on a sample review from the processed dataset to demonstrate how different prompting strategies can lead to vastly different interpretations and outputs from the same model (meta-llama/Llama-3.1-8B-Instruct).Sample Review: 'brings,back,,original,horror,,73'-------------------------
Result from Prompt 1 (Direct):
A very brief review!

The sentiment of this review is: POSITIVE

The reviewer is expressing enthusiasm for something ("brings back original horror") and seems to be referencing a movie or a franchise from the 1970s (mentioned as "73"), implying that it's a nostalgic and enjoyable experience.
-------------------------

Result from Prompt 2 (Role-Playing):
What a delightfully cryptic review!

After some analysis, I'd say this is a BAD review. Here's why:

* The reviewer uses three dots (,,) to separate words, which suggests that they're trying to convey a sense of uncertainty or frustration.
* The phrase "brings back" is typically a positive phrase, but in this context, it's followed by a low score (73) and a somewhat ambiguous sentiment. It implies that the reviewer is not impressed by the movie's ability to bring back the original horror.
* The lack of descriptive language or specific criticisms makes the review feel vague and unhelpful.

Overall, this review seems to suggest that the reviewer was disappointed or underwhelmed by the movie, but they're not providing any concrete reasons why.
This output clearly shows that the direct prompt led to a correct interpretation, while the more complex persona prompt caused the model to "hallucinate" incorrect reasoning, resulting in a failed classification.🚀 How to Run1. Clone the Repositorygit clone <your-repository-url>
cd <your-repository-name>
2. Set Up a Virtual Environment (Recommended)python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
3. Install Dependenciespip install -r requirements.txt
4. Set Up Environment VariablesCreate a .env file in the root directory and add your Hugging Face API token:HUGGINGFACEHUB_API_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"
5. Run the ApplicationExecute the main script to see the sentiment analysis in action:python app.py
🛠️ Tools and Libraries UsedPythonPandas: For data manipulation and preprocessing.NLTK & TextBlob: For NLP cleaning tasks like tokenization, lemmatization, and spelling correction.LangChain: For orchestrating the interaction with the language model and managing prompts.Hugging Face Hub: For accessing the pre-trained language model.Napkin.ai: Used to create the visual flowchart for the final submission.
