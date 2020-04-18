import torch
import torchvision

# Assumes "Bias trick" where last dimension in all tensors is constant 1.
class Linear():
    def __init__(self, D_in, D_out, initialzation):
        self.w = torch.tensor(initialzation(D_in, D_out), requires_grad=True,
                              device='cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, input):
        return input.mm(self.w)

    def params(self):
        return self.w

    def backward(self):
        self.w.zero_grad()
        self.w.backward()


class Net():
    def __init__(self, layers, activation, loss_function):
        self.layers = layers
        self.activation = activation
        self.loss_function = loss_function

    def forward(self, x):
        for layer in self.layers[0, -2]:
            x = layer.forward(x)
            x = self.activation(x)
        return self.layers[-1](x)

    def loss(self, y):
        return self.loss_function(y)

    def backwards(self):
        for layer in self.layers:
            layer.backward()

    def params(self):
        p = []
        for layer in self.layers:
            p.append(layer.params())
        return p


def transformations():
    t = []
    t.append(torch.torchvision.transforms.ToTensor)
    t.append(torch.torchvision.transforms.Lambda(lambda x: x.flatten()))
    t.append(torch.cat([1], out=None)) #Bias trick
    return(torch.torchvision.transforms.Compose(transforms))

