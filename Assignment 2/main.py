import subprocess
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from Apriori import association_rules
from ParameterGrid import ParameterGrid
from Recommendations import recommend_average_confidence, recommend_average_support, recommend_total_confidence, \
    recommend_total_support, recommend_weighted_confidence, recommend_weighted_support, recommend_number_rules, \
    recommend_popularity, recommend_lift
from utils import read_transactions, write_transactions, read_frequent_itemsets, evaluate_recommendations, \
    split_transactions, rules_cache, plot

data_dir = Path("data")
if not data_dir.exists():
    data_dir.mkdir()

lib_dir = Path("lib")
if not lib_dir.exists():
    lib_dir.mkdir()

train_data = data_dir / "train.dat"
test_data = data_dir / "test.dat"

if not train_data.exists() or not test_data.exists():
    transactions = read_transactions(data_dir / "retail.dat")
    train, test = train_test_split(transactions, test_size=0.15)

    write_transactions(data_dir / "train.dat", train)
    write_transactions(data_dir / "test.dat", test)
else:
    train = read_transactions(train_data)
    test = read_transactions(test_data)

test = split_transactions(test)

# min_support = [5, 10, 15, 20, 25, 50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500]
# min_support = [15, 20, 25, 50, 75, 100]
min_support = [25, 50, 75, 100]
# min_support = [25]
# min_confidence = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
min_confidence = [0.01, 0.05, 0.1, 0.2, 0.3]
# top_n = [1, 3, 5, 10, 25]
top_n = [3, 5, 10]
recommendations = [
    recommend_average_confidence,
    recommend_average_support,
    recommend_total_confidence,
    recommend_total_support,
    recommend_weighted_confidence,
    recommend_weighted_support,
    recommend_number_rules,
    recommend_popularity,
    recommend_lift,
]

# Generate ndi and apriori results for each min_support
for support in min_support:
    file = data_dir / "ndi" / f"min_support_{support}.dat"
    if not file.exists():
        subprocess.run([lib_dir / "ndi" / "bf" / "ndi", data_dir / "train.dat", str(support), "100", str(file)],
                       check=True)
    file = data_dir / "apriori" / f"min_support_{support}.dat"
    if not file.exists():
        subprocess.run([lib_dir / "apriori" / "apriori", data_dir / "train.dat", "3", str(support), str(file)],
                       check=True)

param_grid = ParameterGrid({
    'min_support': min_support,
    'ndi': [True, False],
    'top_n': top_n,
    'min_confidence': min_confidence,
    'recommender': recommendations,
})

previous_support = None
previous_ndi = None
previous_confidence = None
frequent_itemsets = None
support_itemsets = None
rules = None
cache = None
support_dict = None
results = dict()
# Run gridsearch for each parameter combination
for params in tqdm(param_grid):
    # print(params)
    ndi = params['ndi']
    support = params['min_support']
    confidence = params['min_confidence']
    top_n = params['top_n']
    if support != previous_support or ndi != previous_ndi:
        file = data_dir / ("ndi" if ndi else "apriori") / f"min_support_{support}.dat"
        frequent_itemsets, support_itemsets = read_frequent_itemsets(file, ndi=ndi)

    if support != previous_support or confidence != previous_confidence or ndi != previous_ndi:
        rules, support_dict = association_rules(frequent_itemsets, support_itemsets, confidence)
        cache = rules_cache(rules)

    recommender = params['recommender']
    recommender_lambda = lambda x: recommender(x, rules, cache, support_dict, top_n)

    key = (top_n, ndi, recommender.__name__)
    support_index = min_support.index(support)
    confidence_index = min_confidence.index(confidence)
    if key not in results:
        results[key] = np.zeros((len(min_support), len(min_confidence), 3))
    results[key][support_index, confidence_index, :] = evaluate_recommendations(test, recommender_lambda, top_n)

    previous_support = support
    previous_ndi = ndi
    previous_confidence = confidence

for metric in ["precision", "recall", "f1_score"]:
    for key, result in results.items():
        if metric == "precision":
            index = 0
        elif metric == "recall":
            index = 1
        elif metric == "f1_score":
            index = 2
        else:
            raise ValueError("Invalid metric")

        plot(metric, key, min_confidence, min_support, result[:, :, index], save=True)
