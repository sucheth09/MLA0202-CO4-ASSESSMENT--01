import pandas as pd
from pgmpy.causal_discovery import HillClimbSearch
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination

data = pd.DataFrame({
    "Age": [
        "Young", "Young", "Young", "Middle",
        "Middle", "Middle", "Old", "Old",
        "Old", "Young", "Middle", "Old"
    ],
    "Income": [
        "Low", "Medium", "High", "Low",
        "Medium", "High", "Low", "Medium",
        "High", "High", "Medium", "High"
    ],
    "VehicleType": [
        "Car", "Bike", "Car", "Bike",
        "Car", "Car", "Bike", "Car",
        "Car", "Bike", "Car", "Bike"
    ],
    "InsuranceClaim": [
        "No", "Yes", "No", "Yes",
        "No", "No", "Yes", "No",
        "No", "Yes", "No", "Yes"
    ]
})

print("Dataset:")
print(data)

search = HillClimbSearch(
    scoring_method="bic-d",
    return_type="dag",
    show_progress=False
)

search.fit(data)

learned_graph = search.causal_graph_

print("\nLearned Network Structure:")
print(list(learned_graph.edges()))

model = DiscreteBayesianNetwork()

model.add_nodes_from(data.columns)

model.add_edges_from(learned_graph.edges())

model.fit(
    data,
    estimator=BayesianEstimator
)

print("\nConditional Probability Tables:")

for cpd in model.get_cpds():
    print(cpd)

inference = VariableElimination(model)

result = inference.query(
    variables=["InsuranceClaim"],
    evidence={
        "Age": "Young",
        "Income": "High",
        "VehicleType": "Bike"
    }
)

print("\nPrediction for New Customer:")
print(result)

states = result.state_names["InsuranceClaim"]

yes_index = states.index("Yes")

claim_probability = result.values[yes_index]

print("\nProbability of Insurance Claim:")
print(claim_probability)
