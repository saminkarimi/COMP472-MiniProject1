import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

def load_knowledge_base(file_path):
    data = pd.read_csv(file_path)
    questions = data["question"].tolist()
    answers = data["answer"].tolist()
    return questions, answers

questions, answers = load_knowledge_base("knowledgebase.csv")

model = SentenceTransformer("all-MiniLM-L6-v2")

question_embeddings = model.encode(questions)

sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)

def find_best_answer(user_question):
    user_embedding = model.encode([user_question])

    similarities = cosine_similarity(
        user_embedding,
        question_embeddings
    )

    best_match_index = similarities.argmax()

    answer = answers[best_match_index]
    similarity_score = similarities[0][best_match_index]

    return answer, similarity_score

def analyze_sentiment(user_question):
    result = sentiment_analyzer(user_question)[0]

    label = result["label"].upper()
    score = result["score"]

    return label, score


print("\nWelcome to Student Support AI")
print("Type 'quit' to exit.\n")

while True:

    user_question = input("Ask a question: ")

    if user_question.lower() == "quit":
        print("Goodbye!")
        break

    label, score = analyze_sentiment(user_question)

    print("\nSentiment:", label)
    print("Confidence Score:", round(score, 2))

    answer, similarity_score = find_best_answer(user_question)

    if label == "NEGATIVE" and score > 0.9:
        print("We recommend contacting a human advisor.")

    print("\nAnswer:")
    print(answer)

    print()
