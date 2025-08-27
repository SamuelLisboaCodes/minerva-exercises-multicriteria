import numpy as np
import pandas as pd

costs = {
    "AWS": {"instance": 35, "db_proc": 210, "db_storage": 120},
    "Azure": {"instance": 30, "db_proc": 200, "db_storage": 115},
    "GCP": {"instance": 32, "db_proc": 190, "db_storage": 110},
}

pairwise = np.array([
    [1,   3,   5,   7],   
    [1/3, 1,   3,   5],   
    [1/5, 1/3, 1,   3],   
    [1/7, 1/5, 1/3, 1]    
])

criteria = ["Custo", "Escalabilidade", "Segurança", "Familiaridade"]

col_sum = pairwise.sum(axis=0)
norm_matrix = pairwise / col_sum
weights = norm_matrix.mean(axis=1)

raw_scores = pd.DataFrame({
    "AWS":  [7, 8, 8, 7],
    "Azure": [8, 9, 8, 6],
    "GCP":  [6, 8, 7, 5],
}, index=criteria)


local_priorities = raw_scores.div(raw_scores.sum(axis=0), axis=1)

global_scores = local_priorities.T.dot(weights)


ranking_base = global_scores.sort_values(ascending=False)

weights_cost = np.array([0.55, 0.25, 0.15, 0.05])
global_scores_cost = local_priorities.T.dot(weights_cost)
ranking_cost = global_scores_cost.sort_values(ascending=False)

print("="*80)
print("Pesos dos critérios (base):")
for c, w in zip(criteria, weights):
    print(f"{c:15s} {w:.3f}")

print("\nRanking (base):")
print(ranking_base)

print("\nRanking (custo priorizado):")
print(ranking_cost)
