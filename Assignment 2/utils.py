import random

import matplotlib.pyplot as plt
import numpy as np


def read_transactions(filename):
    """
    Read transactions from a file
    :param filename: The filename to read from
    :return: A list of sets, where each set is a transaction
    """
    with open(filename) as f:
        transactions = [set(map(int, line.rstrip().split())) for line in f]
    return transactions


def write_transactions(filename, transactions):
    """
    Write transactions to a file
    :param filename: The filename to write to
    :param transactions: A list of sets, where each set is a transaction
    :return: None
    """
    with open(filename, "w") as f:
        f.writelines(map(lambda s: " ".join(map(str, sorted(s))) + "\n", transactions))


def split_transactions(transactions, min_items=10, ground_truth_items=5):
    """
    Split transactions into items to make recommendations for and ground truth items
    :param transactions: The transactions to split
    :param min_items: Minimum number of items in a transaction to be considered
    :param ground_truth_items: Number of items to use as ground truth
    :return: List of tuples, where each tuple is (items to make recommendations for, ground truth items)
    """
    result = list()
    for transaction in transactions:
        if len(transaction) < min_items:
            continue
        ground_truth = set(random.sample(sorted(transaction), ground_truth_items))
        result.append((transaction - ground_truth, ground_truth))
    return result


def read_frequent_itemsets(filename, ndi=False):
    """
    Read frequent itemsets from a file
    :param filename: The filename to read from
    :param ndi: If False, the output of apriori is expected. If True, the output of ndi is expected.
    :return: Frequent itemsets and their supports
    """
    index = -2 if ndi else -1
    with open(filename) as f:
        lines = [line.rstrip().split() for line in f]

    itemsets = [set(map(int, line[:index])) for line in lines]
    supports = [int(line[index][1: -1]) for line in lines]
    # Convert to frozenset
    itemsets = [frozenset(itemset) for itemset in itemsets]
    frequent_itemsets = itemsets
    return frequent_itemsets, supports


def get_popular_items(top_n=10):
    # Get the top n most popular items
    item_counts = {}
    with open("data/retail.dat") as f:
        for line in f:
            for item in line.rstrip().split():
                item_counts[int(item)] = item_counts.get(int(item), 0) + 1
    popular_items = sorted(item_counts, key=item_counts.get, reverse=True)[:top_n]
    return popular_items


def evaluate_recommendations(test_data, recommender_lambda, top_n=5):
    """
    Evaluate a recommender system
    :param test_data: Test data
    :param recommender_lambda: The ranking method to use
    :param top_n: Top n to calculate precision, recall, and F1 score @top_n
    :return: Precision, recall, and F1 score
    """
    true_positives = 0
    # false_positives = 0
    false_negatives = 0

    for input_items, true_items in test_data:
        # Get recommendations for the user
        recommended_items = set(recommender_lambda(input_items))
        true_items = set(true_items)
        true_positives += len(recommended_items.intersection(true_items))
        # false_positives += len(recommended_items - true_items)
        false_negatives += len(true_items - recommended_items)
    # Calculate precision, recall, and F1 score
    precision = true_positives / (len(test_data) * top_n) if len(test_data) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1_score


def rules_cache(rules):
    # Probs to Noah Dani�ls for the idea
    # This cache is used in the recommendations
    # Using this cache we can quickly find the relevant rules for the input items
    # The recommendations will only loop over the relevant rules instead of all rules
    cache = dict()
    for index, (antecedent, _, _, _) in enumerate(rules):
        for item in antecedent:
            if item not in cache:
                cache[item] = set()
            cache[item].add(index)
    return cache


def plot(metric, key, x_labels, y_labels, data, save=True):
    """
    Plot the results of the metrics
    """
    plt.imshow(data, cmap=plt.cm.Greens)

    y_range, x_range = data.shape
    plt.xticks(np.arange(x_range), x_labels)
    plt.yticks(np.arange(y_range), y_labels)
    plt.xlabel("Support")
    plt.ylabel("Confidence")
    plt.title(f"{metric}@{key[0]} for {key[2]} using ndi:{'True' if key[1] else 'False'}")

    for (x, y), value in np.ndenumerate(data):
        t = f"{value:1.3f}"
        plt.text(y, x, t, ha="center", va="center", bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))

    if save:
        plt.savefig(f"results/plots/{metric}/{metric}@{key[0]}_{key[2]}_ndi_{key[1]}.png")
    else:
        plt.show()

