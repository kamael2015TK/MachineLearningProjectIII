import sys 
sys.path.append("./src/helpers")
from standardizer import *
from file_reader import *
from EX_1_1 import runEx_1_1
from EX_1_2 import runEx_1_2
from matplotlib.pyplot import show

data = standardize(getData())

X = data[:, range(0, 7)]
y = data[:, 7].squeeze()

attributeNames = getAttributeNames()
classNames = getClassNames()

print("EX 1.1 Start")
#runEx_1_1(X, attributeNames)
print("EX 1.1 Finished")

print("EX 1.2 Start")
runEx_1_2(X, y)
print("EX 1.2 Finished")

show()