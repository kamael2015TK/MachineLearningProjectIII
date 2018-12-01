from similarity import binarize2
from apyori import apriori
import numpy as np

#
# Janus Bastian Lansner S145349 (85%)
# Duran Köse S147153 (15%)
# 

#from ex12_1_3 import X,labels
# ex12_1_4
# This is a helper function that transforms a binary matrix into transactions.
# Note the format used for courses.txt was (nearly) in a transaction format,
# however we will need the function later which is why we first transformed
# courses.txt to our standard binary-matrix format.
def mat2transactions(X, labels=[]):
    T = []
    for i in range(X.shape[0]):
        l = np.nonzero(X[i, :])[0].tolist()
        if labels:
            l = [labels[i] for i in l]
        T.append(l)
    return T

# apyori requires data to be in a transactions format, forunately we just wrote a helper function to do that.
#T = mat2transactions(X,labels)
#rules = apriori( T, min_support=0.8, min_confidence=1)

# This function print the found rules and also returns a list of rules in the format:
# [(x,y), ...]
# where x -> y
def print_apriori_rules(rules):
    frules = []
    for r in rules:
        for o in r.ordered_statistics:        
            conf = o.confidence
            supp = r.support
            x = ", ".join( list( o.items_base ) )
            y = ", ".join( list( o.items_add ) )
            print("{%s} -> {%s}  (supp: %.3f, conf: %.3f)"%(x,y, supp, conf))
            frules.append( (x,y) )
    return frules

def runEx_3_1(X, attributeNames): 
    Xbin, attributeNamesBin = binarize2(X, attributeNames)
    print("X, i.e. the wine dataset, has now been transformed into:")
    print(Xbin)
    print(attributeNamesBin)
    T = mat2transactions(Xbin,labels=attributeNamesBin)
    rules = apriori(T, min_support=0.35, min_confidence=.8)
    print_apriori_rules(rules)