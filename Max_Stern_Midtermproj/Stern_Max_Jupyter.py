# %% [markdown]
# **<h5>0) Downloading the libraries</h5>**<br>
# Before using the program you need to download the library pandas using 'pip install pandas' and mlxtend using 'pip install mlxtend'

# %% [markdown]
# **<h5>1) Selecting a file and creating the data frame</h5>**<br>
# The program gives the user a list of files to select from and prompts the user to enter the name of one. once the name is entered the program saves the filename and then attempts to open the file. If the file fails to open or isn't found then it prints an error and exits the program. If the file is found then it is formatted appropriately and organized. The file reading and Organization is all done through the pandas library reading the CSV data sheet. The panda reader creates a 10 column data frame, takes each item and puts it into its own column, and creates a row for each transaction.

# %%
import pandas as pan
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
print(csvFile.head())

# %% [markdown]
# **<h5>2) Organizing the data frame</h5>**<br>
# After the transactions are collected from the CSV file and stored into the data frame it is organized using the pandas library. From there each item is put into its own column, if it appears in a transaction its put down as a 1, else its a 0. this happens for all of the items in the datasheet. 

# %%
matrix = \
    pan.get_dummies(csvFile.unstack().dropna()).groupby(level=1).sum()
transactions_count,Item_count =matrix.shape
print(matrix.head())

# %% [markdown]
# **<h5>3) Setting the support and confidence</h5>**<br>
# The user is asked to input the support and confidence using a number between 1 and 100. The number is then turned into a percentage and later used to filter out numbers that are below the given percentage.

# %%
support = input("please enter the minimum support between 1 and 100: ")
conf = input("please enter the confidence as a number between 1 and 100: ")

# %% [markdown]
# **<h5>4) Brute force</h5>**<br>
# To start off, the program starts a timer to see how long it takes to run the program. From there it iterates through every item in the item list and then iterates through every combination of items. Each item is converted to a string and then each combination is saved to a list along with the support for each item set. Next the program iterates through each item and calculates the confidence for the association rules. If there are more than one item in the item set it takes the current item that is being used in the association rule and removed them from the list and adds it to the beginning with an arrow. finally the item is appended to a list with the support and confidence for each item set

# %%
from itertools import combinations
import time
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
for i in range (0,13):
    print("_____________________________")
    print (largeItemsets[i])
    print (patterns[i])



# %% [markdown]
# **<h5>5) Brute force</h5>**<br>
# The program creates a data frame for each frequent item as well as the support using the list for frequent items generated earlier. Afterwards another data frame is created with a column for the item set, the support, and the confidence. using the other list of association rules that were generated. The items in the data frame is then compared the the input support and confidence with the results being saved to a variable. the frequent items data frame is then compared the the user support and saved to another variable.

# %%
freqPat = pan.DataFrame(patterns, columns=["Itemset", "Support"])
freqItemset = pan.DataFrame(largeItemsets, columns=["Itemset", "Support", "Confidence"])
sresults = freqItemset[(freqItemset.Support >= (int(support)/100)) & (freqItemset.Confidence >= (int(conf)/100))]
patresults = freqPat[(freqPat.Support >= (int(support)/100))]

# %% [markdown]
# **<h5>4) Printing</h5>**<br>
# Both of the frequent items and the association rules  data frames are printed out after being converted to a string. The execution time is then printed out at the end.

# %%
print("------------------Frequent items------------------")
print(patresults.to_string())
print("---------------Association Rules---------------------")
print(sresults.to_string())
print("--- %s seconds ---" % (time.time() - start_time))

# %% [markdown]
# **<h5>5) Comparison</h5>**<br>
# the results of the Brute force Apriori program is then compared to the results of apriori library and the fpgrowth library. The time for them is then printed out as well.

# %%
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


