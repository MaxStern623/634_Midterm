import time
import pandas as pan
from itertools import combinations
import os

print("Amazon\nBest buy\nKmart\nTarget\nHomegoods")
File_name = input("please enter the name of the database: ")
file_name = str(os.path.dirname(__file__))+"/"+File_name+'.csv'
abs_path = os.path.abspath(file_name)
try:
    csvFile = pan.read_csv(abs_path,header=None, delimiter=',', engine='python', names=range(10))
except:
    print("ERROR 404: file not found")
    exit()
matrix = \
    pan.get_dummies(csvFile.unstack().dropna()).groupby(level=1).sum()
transactions_count,Item_count =matrix.shape
support = input("please enter the minimum support between 1 and 100: ")
conf = input("please enter the confidence as a number between 1 and 100: ")
start_time = time.time()
largeItemsets = []
patterns = []
for items in range(1, Item_count+1):
    for itemset in combinations(matrix, items):
        I_support = (matrix[list(itemset)].all(axis=1).sum())
        sup = (matrix[list(itemset)].all(axis=1).sum())/20
        s = [str(x) for x in itemset]
        patterns.append([", ".join(s), sup])
        for item in s:
            item_con = matrix[item].value_counts()[1]
            confidence = (I_support / item_con)
            temp = s.copy()
            if(len(s) > 1):
                temp.remove(item)
                temp.insert(0,item+" =>")
            largeItemsets.append([", ".join(temp), sup, confidence])
freqPat = pan.DataFrame(patterns, columns=["Itemset", "Support"])
freqItemset = pan.DataFrame(largeItemsets, columns=["Itemset", "Support", "Confidence"])
sresults = freqItemset[(freqItemset.Support >= (int(support)/100)) & (freqItemset.Confidence >= (int(conf)/100))]
patresults = freqPat[(freqPat.Support >= (int(support)/100))]
print("------------------Frequent items------------------")
print(patresults.to_string())
print("---------------Association Rules---------------------")
print(sresults.to_string())
print("--- %s seconds ---" % (time.time() - start_time))

from mlxtend.frequent_patterns import apriori, association_rules 
Apriori_time = time.time()
frq_items = apriori(matrix, min_support = (int(support)/100),use_colnames = True) 
rules = association_rules(frq_items, metric ="confidence", min_threshold = (int(conf)/100))
rules = rules.sort_values(['support','confidence'], ascending =[False, False])
print(rules[['antecedents','consequents','support','confidence']].to_string())
print("--- Apriori: %s seconds ---" % (time.time() - Apriori_time))

from mlxtend.frequent_patterns import fpgrowth
Fpgrowth_time = time.time()
test= fpgrowth(matrix, min_support=(int(support)/100), use_colnames=True)
rules = association_rules(test, metric ="confidence", min_threshold = (int(conf)/100))
rules = rules.sort_values(['support','confidence'], ascending =[False, False])
print(rules[['antecedents','consequents','support','confidence']].to_string())
print("--- FPgrowth: %s seconds ---" % (time.time() - Fpgrowth_time))
