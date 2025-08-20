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

template_3 = PromptTemplate(
    input_variables=["review_text"],
    template= "You are analyzing sentiment in movie reviews. Below are labeled examples:\n\nExample 1: 'This movie was terrifying but brilliant.' → POSITIVE\nExample 2: 'Too slow and boring, not scary at all.' → NEGATIVE\nExample 3: 'Decent film, but not as good as the original.' → NEUTRAL\n\nNow classify the following review:\n\nReview: {review_text}\n\nSentiment:"
    
)

template_3.save('template_3.json')
#saving prompts into json files to keep the code clean and increase reusability.
