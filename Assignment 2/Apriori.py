import cProfile
import pathlib
import time
import subprocess
from itertools import chain, combinations, filterfalse
from pstats import SortKey
from utils import read_transactions, read_frequent_itemsets


def powerset(iterable):
    """
    Returns the powerset of the iterable.
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def join_set(itemsets, k):
    """
    Joins the itemsets of size k-1 to itemsets of size k.
    :param itemsets: input itemsets
    :param k: size of candidate itemsets
    :return: candidate itemsets of size k
    """
    # Doesn't uses better candidate item generation from slide 25
    return set(
        [itemset1.union(itemset2) for itemset1 in itemsets for itemset2 in itemsets if
         len(itemset1.union(itemset2)) == k]
    )


def itemsets_support(transactions, itemsets, min_support):
    """
    Returns the relative support of the itemsets in the transactions.
    :return: itemsets with relative support >= min_support
    """
    support_count = {itemset: 0 for itemset in itemsets}
    for transaction in transactions:
        for itemset in itemsets:
            if itemset.issubset(transaction):
                support_count[itemset] += 1
    n_transactions = len(transactions)
    return {itemset: support / n_transactions for itemset, support in support_count.items() if
            support / n_transactions >= min_support}


def apriori(transactions, min_support):
    """
    Runs the apriori algorithm
    """
    items = set(chain(*transactions))
    itemsets = [frozenset([item]) for item in items]
    itemsets_by_length = [set()]
    k = 1
    while itemsets:
        support_count = itemsets_support(transactions, itemsets, min_support)
        itemsets_by_length.append(set(support_count.keys()))

        k += 1
        # Changed this line to enforce use of monotonicity
        itemsets = join_set(set(support_count.keys()), k)
    frequent_itemsets = set(chain(*itemsets_by_length))
    return frequent_itemsets, itemsets_by_length


def association_rules(frequent_itemsets, support_itemsets, min_confidence=0.05):
    """
    Returns the association rules from the frequent itemsets.
    """
    support_dict = {itemset: support for itemset, support in zip(frequent_itemsets, support_itemsets)}

    rules = []
    for itemset, support_itemset in zip(frequent_itemsets, support_itemsets):
        for subset in filterfalse(lambda x: not x, powerset(itemset)):
            antecedent = frozenset(subset)
            consequent = itemset - antecedent
            confidence = support_itemset / support_dict[antecedent]
            if confidence >= min_confidence:
                rules.append((antecedent, consequent, support_itemset, confidence))
    return rules, support_dict


if __name__ == "__main__":
    # Example usage
    # transactions = [
    #     {"A", "B", "C"},
    #     {"A", "B"},
    #     {"A", "C"},
    #     {"A"},
    #     {"B", "C"},
    #     {"B"},
    #     {"C"},
    # ]
    start_time = time.time()
    transactions = read_transactions("data/retail.dat")
    min_support = 0.2
    min_confidence = 0.5
    # rules = association_rules(transactions, min_support, min_confidence, cpp=True)
    # for antecedent, consequent, support, confidence in rules:
    #     print(f"{antecedent} => {consequent} (support={support:.2f}, confidence={confidence:.2f})")

    print("--- %s seconds ---" % (time.time() - start_time))

    # Example usage with profiling
    # with cProfile.Profile() as pr:
    #     # ... do something ...
    #     transactions = read_transactions("data/retail.dat")
    #     min_support = 0.2
    #     min_confidence = 0.5
    #     rules = association_rules(transactions, min_support, min_confidence)
    #     for antecedent, consequent, support, confidence in rules:
    #         print(f"{antecedent} => {consequent} (support={support:.2f}, confidence={confidence:.2f})")
    #
    #     pr.print_stats(sort=SortKey.CUMULATIVE)
    #     pr.print_stats(sort=SortKey.TIME)
