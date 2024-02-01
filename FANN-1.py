# -*- coding:utf-8 -*-

from fann2 import libfann

connection_rate = 1
learning_rate = 0.7
input = 2
hidden = 4
output = 1
error = 0.0001
iterations = 100000
reports = 10
ann = libfann.neural_net()
ann.create_sparse_array(connection_rate, (input, hidden, output))
ann.set_learning_rate(learning_rate)
ann.set_activation_function_output(libfann.SIGMOID_SYMMETRIC_STEPWISE)

ann.train_on_file("xor.data", iterations, reports, error)
ann.save("xor.net")



