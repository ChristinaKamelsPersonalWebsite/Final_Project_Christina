from classify_query import classify_query

queries = [
    "give me chest exercises",
    "best leg workout for beginners",
    "how much protein should i eat",
    "what should i eat after training",
    "make me a 4 day workout plan",
    "build me a push pull legs routine",
    "analyze my progress logs",
    "am i getting stronger",
]

print("\nBERT Intent Classification Demo\n")

for q in queries:
    result = classify_query(q)
    print(f"Query: {q}")
    print(f"Predicted label: {result['label']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Scores: {result['scores']}")
    print("-" * 60)