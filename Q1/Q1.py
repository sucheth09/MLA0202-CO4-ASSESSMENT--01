from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

model = DiscreteBayesianNetwork([
    ("Obesity", "Diabetes"),
    ("HighBloodSugar", "Diabetes")
])

cpd_obesity = TabularCPD(
    variable="Obesity",
    variable_card=2,
    values=[[0.7], [0.3]]
)

cpd_sugar = TabularCPD(
    variable="HighBloodSugar",
    variable_card=2,
    values=[[0.6], [0.4]]
)

cpd_diabetes = TabularCPD(
    variable="Diabetes",
    variable_card=2,
    values=[
        [0.99, 0.90, 0.80, 0.30],
        [0.01, 0.10, 0.20, 0.70]
    ],
    evidence=["Obesity", "HighBloodSugar"],
    evidence_card=[2, 2]
)

model.add_cpds(cpd_obesity, cpd_sugar, cpd_diabetes)

print("Model Valid:", model.check_model())

inference = VariableElimination(model)

result = inference.query(
    variables=["Diabetes"],
    evidence={
        "Obesity": 1,
        "HighBloodSugar": 1
    }
)

print("\nProbability of Diabetes:")
print(result)
