from sklearn.model_selection import train_test_split

from Apriori import association_rules, read_transactions
from utils import get_popular_items

popular_items = get_popular_items(50)


def recommend_average_confidence(input_items, rules, cache, support_dict, top_n=5):
    relevant_rules = set()
    for item in input_items:
        if item in cache:
            relevant_rules |= cache[item]

    recommendations = {}
    for index in relevant_rules:
        antecedent, consequent, support, confidence = rules[index]
        if antecedent.issubset(input_items):
            for item in consequent - input_items:
                if item not in recommendations:
                    recommendations[item] = []
                recommendations[item].append((confidence, support))
    recommendations = {
        item: (
            sum(conf for conf, _ in item_rules) / len(item_rules), sum(sup for _, sup in item_rules) / len(item_rules))
        for item, item_rules in recommendations.items()
    }
    sorted_recommendations = sorted(recommendations.items(), key=lambda x: (-x[1][0], -x[1][1]))
    return [item for item, _ in sorted_recommendations[:top_n]]


def recommend_average_support(input_items, rules, cache, support_dict, top_n=5):
    relevant_rules = set()
    for item in input_items:
        if item in cache:
            relevant_rules |= cache[item]

    recommendations = {}
    for index in relevant_rules:
        antecedent, consequent, support, confidence = rules[index]
        if antecedent.issubset(input_items):
            for item in consequent - input_items:
                if item not in recommendations:
                    recommendations[item] = []
                recommendations[item].append((confidence, support))
    recommendations = {
        item: (
            sum(conf for conf, _ in item_rules) / len(item_rules), sum(sup for _, sup in item_rules) / len(item_rules))
        for item, item_rules in recommendations.items()
    }
    sorted_recommendations = sorted(recommendations.items(), key=lambda x: (-x[1][1], -x[1][0]))
    return [item for item, _ in sorted_recommendations[:top_n]]


def recommend_total_confidence(input_items, rules, cache, support_dict, top_n=5):
    relevant_rules = set()
    for item in input_items:
        if item in cache:
            relevant_rules |= cache[item]

    recommendations = {}
    for index in relevant_rules:
        antecedent, consequent, support, confidence = rules[index]
        if antecedent.issubset(input_items):
            for item in consequent - input_items:
                if item not in recommendations:
                    recommendations[item] = 0
                recommendations[item] += confidence
    sorted_recommendations = sorted(recommendations.items(), key=lambda x: -x[1])
    return [item for item, _ in sorted_recommendations[:top_n]]


def recommend_total_support(input_items, rules, cache, support_dict, top_n=5):
    relevant_rules = set()
    for item in input_items:
        if item in cache:
            relevant_rules |= cache[item]

    recommendations = {}
    for index in relevant_rules:
        antecedent, consequent, support, confidence = rules[index]
        if antecedent.issubset(input_items):
            for item in consequent - input_items:
                if item not in recommendations:
                    recommendations[item] = 0
                recommendations[item] += support
    sorted_recommendations = sorted(recommendations.items(), key=lambda x: -x[1])
    return [item for item, _ in sorted_recommendations[:top_n]]


def recommend_weighted_confidence(input_items, rules, cache, support_dict, top_n=5):
    relevant_rules = set()
    for item in input_items:
        if item in cache:
            relevant_rules |= cache[item]

    recommendations = {}
    for index in relevant_rules:
        antecedent, consequent, support, confidence = rules[index]
        if antecedent.issubset(input_items):
            weight = 1 + len(consequent.intersection(input_items))
            for item in consequent - input_items:
                if item not in recommendations:
                    recommendations[item] = 0
                recommendations[item] += weight * confidence
    sorted_recommendations = sorted(recommendations.items(), key=lambda x: -x[1])
    return [item for item, _ in sorted_recommendations[:top_n]]


def recommend_weighted_support(input_items, rules, cache, support_dict, top_n=5):
    relevant_rules = set()
    for item in input_items:
        if item in cache:
            relevant_rules |= cache[item]

    recommendations = {}
    for index in relevant_rules:
        antecedent, consequent, support, confidence = rules[index]
        if antecedent.issubset(input_items):
            weight = 1 + len(consequent.intersection(input_items))
            for item in consequent - input_items:
                if item not in recommendations:
                    recommendations[item] = 0
                recommendations[item] += weight * support
    sorted_recommendations = sorted(recommendations.items(), key=lambda x: -x[1])
    return [item for item, _ in sorted_recommendations[:top_n]]


def recommend_number_rules(input_items, rules, cache, support_dict, top_n=5):
    relevant_rules = set()
    for item in input_items:
        if item in cache:
            relevant_rules |= cache[item]

    recommendations = {}
    for index in relevant_rules:
        antecedent, consequent, support, confidence = rules[index]
        if antecedent.issubset(input_items):
            for item in consequent - input_items:
                if item not in recommendations:
                    recommendations[item] = 0
                recommendations[item] += 1
    sorted_recommendations = sorted(recommendations.items(), key=lambda x: -x[1])
    return [item for item, _ in sorted_recommendations[:top_n]]


def recommend_popularity(input_items, rules, cache, support_dict, top_n=5):
    recommendations = list()
    for item in popular_items:
        if item not in input_items:
            recommendations.append(item)
    return recommendations[:top_n]


def recommend_lift(input_items, rules, cache, support_dict, top_n=5):
    relevant_rules = set()
    for item in input_items:
        if item in cache:
            relevant_rules |= cache[item]

    recommendations = {}
    for index in relevant_rules:
        antecedent, consequent, support, confidence = rules[index]
        if antecedent.issubset(input_items):
            for item in consequent - input_items:
                if item not in recommendations:
                    recommendations[item] = 0
                recommendations[item] += confidence / support_dict[antecedent]
    sorted_recommendations = sorted(recommendations.items(), key=lambda x: -x[1])
    return [item for item, _ in sorted_recommendations[:top_n]]


if __name__ == "__main__":
    # input_items = {"A", "B"}
    # transactions = [
    #     {"A", "B", "C"},
    #     {"A", "B"},
    #     {"A", "C"},
    #     {"A"},
    #     {"B", "C"},
    #     {"B"},
    #     {"C"},
    # ]
    transactions = read_transactions("data/retail.dat")
    x_train, x_test = train_test_split(transactions, test_size=0.5)
    input_items = {39}
    min_support = 0.2
    min_confidence = 0.5
    # rules = association_rules(x_train, min_support, min_confidence, True)
    # recommended_items = recommend_average_confidence(input_items, rules)
    # print("Recommended items:", recommended_items)
