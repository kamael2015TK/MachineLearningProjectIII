import sys 
sys.path.append("./src/helpers")
from standardizer import *
from file_reader import *
from EX_1_1 import runEx_1_1
from EX_1_2 import runEx_1_2
from EX_2_1 import runEx_2_1
from EX_3_1 import runEx_3_1
from matplotlib.pyplot import show

data = standardize(getData())

X = data[:, range(0, 7)]
y = data[:, 7].squeeze()

attributeNames = getAttributeNames()[0:7]
classNames = getClassNames()

print("EX 1.1 Start")
runEx_1_1(X, attributeNames, getData()[:,7])
print("EX 1.1 Finished")

print("EX 1.2 Start")
runEx_1_2(X, y, getData()[:,7], len(attributeNames), classNames)
print("EX 1.2 Finished")

print("EX 2.1 Start")
runEx_2_1(data)
print("EX 2.1 Finished")

print("EX 3.1 Start")
runEx_3_1(data, getAttributeNames())
print("EX 3.1 Finished")

show()