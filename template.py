from langchain_core.prompts import PromptTemplate


template_1 = PromptTemplate(
    input_variables=["review_text"],
    template="Sentiment for this review: '{review_text}'"
)
template_1.save('template_1.json')



template_2 = PromptTemplate(
    input_variables=["review_text"],
    template="As a horror fan, is this a good or bad review? Review: '{review_text}'"
)
template_2.save('template_2.json')

#saving prompts into json files to keep the code clean and increase reusability.
