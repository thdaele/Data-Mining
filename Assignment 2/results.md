# About profiling_apriori

We can see that 90s is spend in calculating the issubset and 170s is spend in the itemsets_support function from which 90s is the issubset.
So the resting 80s is spend in the double loop and incrementing the support_count.
Which is exactly the following lines from the pseudocode from class.
```
    for each transaction t in database do
    increment the count of all candidates in C k+1
    that are contained in t
```
```Python
for transaction in transactions:
    for itemset in itemsets:
        if itemset.issubset(transaction):
            support_count[itemset] += 1 
```