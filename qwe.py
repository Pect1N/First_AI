from array import array
from cmath import sqrt
from operator import mod, truth
import numpy
import scipy.special

class neuralNetwork:
    def __init__(self, inputnodes, hidennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hidennodes
        self.onodes = outputnodes
        self.lr = learningrate
        self.wih = (numpy.random.rand(self.hnodes, self.inodes) - 0.5)
        self.who = (numpy.random.rand(self.onodes, self.hnodes) - 0.5)
        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self, inputs_list, targets_list):
        inputs = numpy.array(inputs_list, ndmin = 2).T
        targets = numpy.array(targets_list, ndmin = 2).T

        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors)

        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1 - final_outputs)), numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)), numpy.transpose(inputs))

    def query(self, inputs_list, real_value):
        inputs = numpy.array(inputs_list, ndmin = 2).T
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        max_value = max(final_outputs)[0]
        result = numpy.argmax(final_outputs)

        qwe = numpy.asarray(final_outputs)
        summ_test = qwe.sum()
        summ = 0
        answers = [0, 0]
        for i in final_outputs:
            summ += i[0]
        if int(real_value) == result:
            answers[1] += 1
            text = "True"
        else:
            answers[0] += 1
            text = "False"
        print(real_value, result)
        # print(final_outputs)
        print(round(max_value, 4), round((max_value * 100) / summ, 2), "%")
        print(text, "\n")
        return answers

input_nodes = 784
hiden_nodes = 100
output_nodes = 10

neural_number = 1

l_r = [0.9, 0.7, 0.5, 0.3, 0.1]

n = list()
for i in range(neural_number):
    n.append(neuralNetwork(input_nodes, hiden_nodes, output_nodes, l_r[i]))

data_file = open("mnist_dataset\mnist_train.csv", 'r')
data_list = data_file.readlines()
data_file.close()

i = 0
procent = 1
for record in data_list:
    if i == 0: print("start training")
    elif i == int(len(data_list) * (procent / 100)):
        if procent % 10 == 0: print(procent, "%", end = '', sep = '')
        else: print(".", end = '', sep = '')
        procent += 1
    elif i == len(data_list): print("\ntraining done")
    i += 1

    all_values = record.split(',')
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    targets = numpy.zeros(output_nodes) + 0.01
    targets[int(all_values[0])] = 0.99
    for j in range(neural_number):
        n[j].train(inputs, targets)

test_data_file = open("mnist_dataset\mnist_test_10.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

for i in range(neural_number):
    answ = [0, 0]
    wrong = 0
    right = 0
    for testing in test_data_list:
        all_values = testing.split(',')
        answ = n[i].query((numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01, all_values[0])
        wrong += answ[0]
        right += answ[1]
    print("For learning rate", l_r[i])
    print("Number of wrong ansvers", wrong, "/ 10")
    print("Number of right ansvers", right, "/ 10")