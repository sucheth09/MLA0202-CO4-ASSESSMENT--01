import numpy as np
from hmmlearn import hmm

states = ["Sunny", "Cloudy", "Rainy"]
observations = ["Dry", "Damp", "Wet"]

model = hmm.CategoricalHMM(
    n_components=3,
    random_state=42
)

model.startprob_ = np.array([
    0.5,
    0.3,
    0.2
])

model.transmat_ = np.array([
    [0.6, 0.3, 0.1],
    [0.2, 0.5, 0.3],
    [0.1, 0.3, 0.6]
])

model.emissionprob_ = np.array([
    [0.7, 0.2, 0.1],
    [0.3, 0.5, 0.2],
    [0.1, 0.3, 0.6]
])

observation_sequence = ["Dry", "Damp", "Wet", "Wet", "Damp"]

observation_codes = {
    "Dry": 0,
    "Damp": 1,
    "Wet": 2
}

X = np.array([
    [observation_codes[x]]
    for x in observation_sequence
])

log_probability, hidden_states = model.decode(
    X,
    algorithm="viterbi"
)

predicted_states = [
    states[i]
    for i in hidden_states
]

print("Observed Sequence:")
print(observation_sequence)

print("\nPredicted Hidden Weather States:")
print(predicted_states)
