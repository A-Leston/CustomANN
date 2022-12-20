# Artificial neural network for binary classification by ariel leston
import math                                                                # used for value of e in sigmoid function calculation
import random                                                              # used for random initialization of weights
import time                                                                # used for timing everything


class Node:                                                                # Node class, expects to be initialized with locational data for where it is in neural net
    def __init__(self, ids):                                               # Ids = (overallNodeNum, (layerNum, nodeNum on layer))
        self.nodeId, self.position = ids                                   # splitting IDs into its 2 parts and saving them, the node number and node position
        self.inputs, self.outputs = [], []                                 # preparing lists for input nodes and output nodes that will link to this node
        self.a, self.delta = None, None                                    # preparing place holders for node activation value and delta value (delta = error)
        self.bias = 1                                                      # initializing node bias value (bias is a node weight)

    def __str__(self):                                                     # print function for individual nodes
        return f"node {self.nodeId} at position {self.position} with an outValue of {self.a}"


class Model:                                                               # Model class, expects to get a nodeMap to represent number of nodes and layers ex. [2,3,1]
    def __init__(self, nodeMap, baseWeight=None):                          # baseWeight is optional, if given will be used as starting value for all path weights
        self.nodes, self.layers, self.paths = [], [], []                   # preparing node list (reused for each layer), layers list, and paths list
        self.nodeNum, self.maxDiff = 1, 1                                  # set starting values for node num and max difference
        self.loops = 0                                                     # set starting value for loop count, part of stop check in train
        self.baseWeight = baseWeight                                       # set starting value for path weights, if none then is randomly generated
        self.times = [0, 0, 0, 0]                                          # time spent while in [activation, backPropagate, weightUpdate, overall training]
        self.createNet(nodeMap)                                            # setup neural net, which is a list of layers where each layer is a list of node objects

    def createNet(self, nodeMap):
        for h in range(len(nodeMap)):                                      # for each layer in the nodeMap instructions (ex.[2,3,1])
            for i in range(nodeMap[h]):                                    # then for each node that is going to be in that layer
                position = (h, i)                                          # create that nodes position tuple (layerNum, nodeNum on layer)
                nodeIds = (self.nodeNum, position)                         # set up node IDs = (overallNodeNum, (layerNum, nodeNum on layer))
                self.nodes.append(Node(nodeIds))                           # create this node and add it to node list (which will be added to layers list later)
                self.nodeNum += 1                                          # keeping track of overall node count
                if h > 0:                                                  # while creating the nodes and layers, also going to set up the paths list
                    for x in range(nodeMap[h-1]):                          # if this node is not on first layer, connect paths from current node to nodes on previous layer
                        self.layers[h - 1][x].outputs.append(self.nodes[i])  # connect nodes by putting current node on output list of a node on the layer above it
                        self.nodes[i].inputs.append(self.layers[h-1][x])   # and then must also put that node on current nodes input list
                        if not self.baseWeight:                            # each path in paths list is (node1, node2, pathWeight) where node1 feeds into node2
                            self.paths.append([self.layers[h - 1][x], self.nodes[i], random.randint(1, 1500) / 1000])   # if base weight is none, get a random one for this path
                        else:                                              # otherwise, just use the given weight for each path
                            self.paths.append([self.layers[h - 1][x], self.nodes[i], self.baseWeight])
            self.layers.append(self.nodes)                                 # add current node list to the layers list as a layer
            self.nodes = []                                                # then reset the node list for next loop's layer

    def activate(self, xlist):                                             # to push values through all nodes activation function
        startT1 = time.time()
        x, w = 0, 0                                                        # using x for xlist index, xlist is x value(s) in single datapoint, and w as paths list index
        for layer in self.layers:                                          # loop through all layers and all nodes on each layer
            for node in layer:
                if node.position[0] == 0:                                  # if node on the input layer (first layer), just pass value through, no sigmoid activation
                    node.a = xlist[x]
                else:                                                      # every other node will sum up all the inputs from its paths (input value * that paths weight)
                    sumInVal = 0                                           # will use that sum as its input for its own activation function, which sets its output value
                    for inputNode in node.inputs:
                        sumInVal += inputNode.a * self.paths[w][2]
                        w += 1
                    node.a = 1 / (1 + (math.e ** -(sumInVal + node.bias)))  # then do sigmoid activation of this node
                x += 1
        self.times[0] += time.time() - startT1

    def backPropagate(self, y):                                           # to back propagate error through all nodes using the true answer/class
        startT2 = time.time()                                             # will need to get output layers error first then move upward (last layer first)
        self.layers.reverse(); self.paths.reverse()                       # so reversing layers and paths to go backwards easily, will undo later
        i, x = 0, 0                                                       # using i as layer index, x as paths list index
        for layer in self.layers:                                         # loop through each layer in the net (backwards or bottom up since I reversed)
            if i == 0:                                                    # if its an output layer, calc and set the delta value for each of those nodes (diff calc)
                for endNode in layer:
                    endNode.delta = (y - endNode.a) * (endNode.a * (1 - endNode.a))
            elif 0 < i < len(self.layers) - 1:                            # else if this is a hidden layer (not input nor output layers)
                for node in layer:                                        # loop through each node on that layer
                    sumVal = 0
                    for output in node.outputs:                           # and loop through each node that this node outputs to (its following nodes)
                        sumVal += self.paths[x][2] * output.delta         # sum up the (weight * error) of each path
                        x += 1
                    node.delta = (node.a * (1 - node.a)) * sumVal         # calc and set delta value (error) for that node
            i += 1
        self.layers.reverse(); self.paths.reverse()                       # undoing the earlier reverse by reversing again
        self.times[1] += time.time() - startT2

    def weightUpdate(self, lr):                                           # to update all the path weights and node biases based on node delta values (based on the error)
        startT3 = time.time()
        self.diffs = []                                                   # setup differences list used for stopping training loop (when change in error small enough)
        for path in self.paths:                                           # loop through each node path in the paths list
            inNode, outNode, weight = path[0], path[1], path[2]           # separating the paths parts for clarity
            newWeight = weight + (lr * inNode.a * outNode.delta)          # calc the new path weight
            self.diffs.append(abs(newWeight - weight))                    # save the difference between new weight and old weight, used for check to stop training
            path[2] = newWeight                                           # set the new weight on that path
        for layer in self.layers[1:]:                                     # then need to loop through all layers and all nodes to update node biases (except first layer)
            for node in layer:
                newBias = node.bias + (lr * node.delta)                   # calc the new bias value
                self.diffs.append(abs(newBias - node.bias))               # save the difference between old and new in diffs list, used for stop check in train
                node.bias = newBias                                       # set the new bias on this node
        self.times[2] += time.time() - startT3

    def train(self, train, target, lr=0.15, maxLoops=10**5, convg=10**-5):  # to train model (updating weights until the error difference small enough to say its converged)
        startT4 = time.time()
        self.loops = 0
        while self.loops < maxLoops and self.maxDiff > convg:            # need to loop until difference between weights is less than convg or until loop limit is hit
            for x in range(len(train)):                                  # for each set of data points or coordinates in training set
                self.activate(train[x])                                  # call activate with that data, pushing the values through all the nodes
                self.backPropagate(target[x])                            # then back propagate the error to fill delta values
                self.weightUpdate(lr)                                    # then update weights using new deltas(error), also fills diffs list inside of this function
                self.maxDiff = max(self.diffs)                           # then need to use max or average of all diffs for converge check
            self.loops += 1
        self.times[3] += time.time() - startT4

    def test(self, xlist):
        self.activate(xlist)                                             # to test model only need to do activate, not training so no back propagate or weight updates
        return self.layers[-1][-1].a                                     # after activating the guess is just whatever value is coming out of last node, so return that

    def __str__(self):
        retString = "---Model structure---\n"                            # print model structure by layers
        for layer in self.layers:
            for node in layer:
                retString += f"node {node.nodeId} at {node.position}\n     bias = {node.bias}, delta = {node.delta} \n     outVal = {node.a} \n"
        retString += "-------paths-------\n"                             # print model structure by node connections
        for node in self.paths:
            retString += f"node {node[0].nodeId} --({node[2]})--> node {node[1].nodeId}\n"
        return retString
