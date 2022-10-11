import torch
import torch.nn as nn


def build_mlp(input_size, output_size, n_layers, size, activation= nn.Tanh, output_activation=None):
    """
        #inputs:
            input_size: dimension of inputs.
            output_size: dimension of outputs.
            n_layers: number of the layers in the Sequential model.
            size: width of each hidden layer.(for the sake of simplicity, we take the width of all the hidden layers the same)
            activation: activation function used after each hidden layer.
            output_activation: activation function used in the last layer.

        #outputs:
            model: the implemented model.
    """
    # TODO: Sequentially append the layers to the list.
    # TODO: Use Xavier initialization for the weights.
    # Hint: Look at nn.Linear and nn.init.

    layers = []
    layer_temp = nn.Linear(input_size,size)
    gain = nn.init.calculate_gain('tanh',param=None)
    nn.init.xavier_uniform_(layer_temp.weight,gain)
    layers.append(layer_temp)
    layers.append(activation())
    for i in range(n_layers):
        layer_temp = nn.Linear(size,size)
        gain = nn.init.calculate_gain('tanh', param=None)
        nn.init.xavier_uniform_(layer_temp.weight, gain)
        layers.append(layer_temp)
        layers.append(activation())
    layer_temp = nn.Linear(size,output_size)
    gain = 1
    nn.init.xavier_uniform_(layer_temp.weight,gain)
    layers.append(layer_temp)
    model = nn.Sequential(*layers)

    return model

############################################
############################################

device= torch.device("cuda")
dtype= torch.float32

def lrelu(x, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * abs(x)
