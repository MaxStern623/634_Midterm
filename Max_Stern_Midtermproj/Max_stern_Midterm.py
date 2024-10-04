import pathlib
import time
import pandas as pan
from itertools import combinations
from pathlib import Path
print(pathlib.Path().resolve())
print(pathlib.Path().absolute())
import os
#file_name = str(pathlib.Path().resolve())+"/Amazon.csv"
#abs_path = os.path.abspath(file_name)
#print(abs_path)

print("Amazon\nBest buy\nKmart\nTarget\nHomegoods")
File_name = input("please enter the name of the database: ")
file_name = str(pathlib.Path().resolve())+"/"+File_name+'.csv'
abs_path = os.path.abspath(file_name)
print(abs_path)
#print(str(Path.cwd())+'\\'+File_name+'.csv')
try:
    csvFile = pan.read_csv(Path.cwd()+File_name+'.csv',header=None, delimiter=',', engine='python', names=range(10))
except:
    print("ERROR 404: file not found")
    exit()
matrix = \
    pan.get_dummies(csvFile.unstack().dropna()).groupby(level=1).sum()
transactions_count,Item_count =matrix.shape
#print(type(matrix))
#print('transactions: ',transactions_count)
#print('items: ',Item_count,'\n')
#print(matrix)
support = input("please enter the minimum support between 1 and 100: ")
conf = input("please enter the confidence as a number between 1 and 100: ")
start_time = time.time()
largeItemsets = []
patterns = []
for items in range(1, Item_count+1):
    #count = True
    for itemset in combinations(matrix, items):
        #print("ITEMSET: "+str(itemset))
        I_support = (matrix[list(itemset)].all(axis=1).sum())
        sup = (matrix[list(itemset)].all(axis=1).sum())/20
        s = [str(x) for x in itemset]
        #print(s)
        #if (len(s) >= 1):
        #count =0
        patterns.append([", ".join(s), sup])
        for item in s:
            #count+=1
            #print(len(item))
                #print(len(s))
            item_con = matrix[item].value_counts()[1]
            confidence = (I_support / item_con)
                #item_con = 100
                #print(item_con)
            #if(s.count(',') >= 1):
                #print("bing")
            #item_con = matrix[con].all(axis=1).sum()
            temp = s.copy()
            if(len(s) > 1):
                    #print(temp.index(item))
                temp.remove(item)
                temp.insert(0,item+" =>")
                    #print(count)
                    #print()
            largeItemsets.append([", ".join(temp), sup, confidence])
            #print(type(largeItemsets))
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