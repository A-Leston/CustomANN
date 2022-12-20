from CustomANN import Model
from sklearn import datasets

origData = datasets.load_iris()
xValues = origData.data.tolist()
yValues = origData.target.tolist()
print("all x values: ", xValues)
print("all y values:", yValues)

trainData = []
trainTarget = []
testData = []
testClasses = []

trainSplit = 0.8                                       # the % of data to use for training, left over is for testing
splitSpot = int(50*trainSplit)

# Class 0 data separation (SETOSA flowers)
for i in range(0, splitSpot):                           # adding class 0 to train data and train target
    trainData.append(xValues[i])
    trainTarget.append(yValues[i])
for i in range(splitSpot, 50):                          # adding class 0 to test data and test classes
    testData.append(xValues[i])
    testClasses.append(yValues[i])

# Class 1 data separation   (VERSICOLOR flowers)
for i in range(50, 50 + splitSpot):                     # adding class 1 to train data and train target
    trainData.append(xValues[i])
    trainTarget.append(yValues[i])
for i in range(50 + splitSpot, 100):                    # adding class 1 to test data and test classes
    testData.append(xValues[i])
    testClasses.append(yValues[i])

# not going to use the third classes data as this Model is currently only set up to output a single value between 0 and 1. so can only have 2 classes or a True/False structure
# (**it IS possible to use more but would have to add it to the true or false cases in targets list, so instead of this class being 2 as target, it would be added to 0 or 1 class)

net = Model(nodeMap=[4, 3, 1])                        # nodeMap = [input layer, ..., hidden layer(s), ..., output layer]
#print("\nmodel node values and path weights before training:\n", net)
print("training...")                                  # input layer size CANT exceed number of x features, and output layer should always be 1 as model is for binary classification
net.train(trainData, trainTarget, lr=0.175, maxLoops=10**4, convg=10**-5)  # for train, default lr is 0.15, default maxloops 10**5, default convg 10**-5
print(f"training took {net.loops} loops")
print(f"training times in seconds:"
      f"\n  activation={net.times[0]}, back propagate={net.times[1]}, weight update={net.times[2]}"
      f"\n  overhead={net.times[3] - (net.times[0] + net.times[1] + net.times[2])}, total={net.times[3]}")
print("\nmodel node values and path weights after training:\n", net)
print("now testing with test set..")
correct = 0
for x in range(len(testData)):
    guess = net.test(testData[x]).__round__()                   # test takes one set of x values and guesses the corresponding y value (must round to check for equality)
    print(f"test data = {testData[x]}, model guess = {guess}, correct class = {testClasses[x]}")
    if guess == testClasses[x]:
        correct += 1
accuracy = correct / len(testClasses) * 100
print(f"\noverall accuracy was: {accuracy}% ")
