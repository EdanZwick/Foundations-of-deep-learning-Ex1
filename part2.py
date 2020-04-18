import torch
import torchvision


class Flatten:
    def __call__(self, sample):
        return torch.flatten(sample).cuda() if torch.cuda.is_available() else torch.flatten(sample)


def get_transformation():
    return torchvision.transforms.Compose([torchvision.transforms.ToTensor(), Flatten()])


class OneHot:
    def __call__(self, value):
        y = torch.zeros(10)
        y[value] = 1
        return y


class Linear:
    def __init__(self, D_in, D_out, std):
        self.w = torch.empty((D_in + 1, D_out), dtype=torch.float).normal_(mean=0, std=std)
        if torch.cuda.is_available():
            self.w = self.w.cuda()
            self.device = torch.device('cuda')
        self.w.requires_grad = True

    def forward(self, x):
        ones = torch.ones(x.shape[0], 1, device=self.device)
        x = torch.cat((x, ones), dim=1)  # bias trick, avoids keeping track of two parameters.
        return x.mm(self.w)

    def params(self):
        return self.w


class Net:
    def __init__(self, din, dout, dhidden, std):
        self.layers = []
        for dim in dhidden:
            self.layers.append(Linear(din, dim, std))
            din = dim
        self.output = Linear(din, dout, std)  # output  layer

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
            torch.nn.functional.relu(x, inplace=True)
        return self.output.forward(x)

    def params(self):
        p = []
        for layer in self.layers:
            p.append(layer.params())
        return p


def grid_search(train_loader, validation_loader, learning_rates, momentum, in_dim, out_dim, hidden_dim):
    best_loss = 10000000
    for lr in learning_rates:
        for m in momentum:
            net = Net(in_dim, out_dim, hidden_dim, std=1)
            optimizer = torch.optim.SGD(net.params(), lr=lr, momentum=m)
            train(net, train_loader, optimizer)
            validation_loss = run(net, validation_loader)
            print('For parameters: lr-{}, momentum-{} got loss: {}'.format(lr, m, validation_loss))
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_params = {'lr': lr, 'momentum': m}
    return best_params, best_loss


def train(net, loader, optimizer, verbose=False, num_epochs=100):
    for epoch in range(num_epochs):
        epoch_loss = 0
        for i, (batch, target) in enumerate(loader):
            optimizer.zero_grad()
            output = net.forward(batch)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss * batch.shape[0]  # avg. weighted per sample, as not all batches are equal
        if verbose:
            print('Epoch: {}, loss {}'.format(epoch, epoch_loss/len(loader)))
        if epoch % 10 == 0:
            print('epoch: {}'.format(epoch))


def run(net, loader):
    loss = 0
    for i, (batch, target) in enumerate(loader):
        output = net.forward(batch)
        loss += torch.nn.functional.binary_cross_entropy_with_logits(output, target) * batch.shape[0]
    return loss / len(loader)
