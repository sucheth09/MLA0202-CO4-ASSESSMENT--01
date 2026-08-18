import sklearn_crfsuite

def word_features(sentence, i):
    word = sentence[i]

    features = {
        "word.lower()": word.lower(),
        "word.isupper()": word.isupper(),
        "word.istitle()": word.istitle(),
        "word.isdigit()": word.isdigit(),
        "word.length": len(word)
    }

    if i > 0:
        features["previous_word"] = sentence[i - 1].lower()
    else:
        features["BOS"] = True

    if i < len(sentence) - 1:
        features["next_word"] = sentence[i + 1].lower()
    else:
        features["EOS"] = True

    return features

def sentence_features(sentence):
    return [
        word_features(sentence, i)
        for i in range(len(sentence))
    ]

X_train = [
    ["John", "ordered", "Laptop", "with", "Order123"],
    ["Priya", "bought", "iPhone", "Order456"],
    ["Rahul", "requested", "Headphones", "under", "Order789"],
    ["Anita", "ordered", "Tablet", "Order321"]
]

y_train = [
    ["B-CUSTOMER", "O", "B-PRODUCT", "O", "B-ORDER_ID"],
    ["B-CUSTOMER", "O", "B-PRODUCT", "B-ORDER_ID"],
    ["B-CUSTOMER", "O", "B-PRODUCT", "O", "B-ORDER_ID"],
    ["B-CUSTOMER", "O", "B-PRODUCT", "B-ORDER_ID"]
]

X = [
    sentence_features(sentence)
    for sentence in X_train
]

crf = sklearn_crfsuite.CRF(
    algorithm="lbfgs",
    max_iterations=100,
    all_possible_transitions=True
)

crf.fit(X, y_train)

test_sentence = [
    "Kiran",
    "ordered",
    "Laptop",
    "with",
    "Order999"
]

X_test = [
    sentence_features(test_sentence)
]

prediction = crf.predict(X_test)

print("Input Sentence:")
print(" ".join(test_sentence))

print("\nPredicted Labels:")

for word, label in zip(test_sentence, prediction[0]):
    print(word, "->", label)
