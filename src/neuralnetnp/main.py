import network
import loaddata.load as load
import datawrapper

"""Based on instructions by Nielsen from http://neuralnetworksanddeeplearning.com/
Uses dataset from Nielsen and dataset obtained via load.load_mnist() """

# Our own dataset
training_own, test_own = load.load_mnist()
training_own, test_own = datawrapper.data_wrapper(training_own, test_own)

#Nielsens Dataset
training_data, validation_data, test_data = load.load_data_wrapper()

# cut data to save computation time if necessary
"""
training_own = training_own[:20000]
test_own = test_own[:1000]
training_data = training_data[:20000]
test_data = test_data[:1000]
"""

#initialize Networks
net = network.Network([784, 30, 20, 10])
net_own = network.Network([784, 30, 20, 10])

# Train networks
print("Training a network on Nielsen's dataset")
net.SGD(training_data, 3, 10, 3.0, test_data=test_data)
print("Training a network on our data")
net_own.SGD(training_own, 3, 10, 1.0, test_data=test_own)
