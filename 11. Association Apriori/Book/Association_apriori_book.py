#Implementing Apriori algorithm from mlxtend

#Conda install mlxtend

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules



book =pd.read_csv("C:\\Users\\user\\Desktop\\data Sc with R & Pyhton\\book.csv")




from mlxtend.preprocessing import TransactionEncoder

te = TransactionEncoder()
te_ary = te.fit(book).transform(book)
book1 = pd.DataFrame(te_ary, columns=te.columns_)

from mlxtend.frequent_patterns import apriori
item = apriori(book1, min_support=0.000000001, max_len=3)
item1 =apriori(book1, min_support=0.000000001, use_colnames=True)

frequent_itemsets = apriori(book1, min_support=0.000000001, use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
frequent_itemsets

















