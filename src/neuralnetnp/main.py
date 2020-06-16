import loaddata.load as load
import network
import datawrapper


"""Based on neural-networks-and-deep-learning by Nielsen"""
training_data, test_data = load.load_mnist()

training_data, test_data = datawrapper.data_wrapper(training_data, test_data)

training_data = list(training_data)
test_data = list(test_data)

net = network.Network([784, 30, 20, 10])

print("The end")

net.SGD(training_data, 5, 20, 3.0, test_data=test_data)
