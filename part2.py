import torch
import torchvision
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

class Flatten:
    def __call__(self, sample):
        return torch.flatten(sample)


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
        self.w.requires_grad = True

    def forward(self, x):
        ones = torch.ones(x.shape[0], 1).cuda() if torch.cuda.is_available() else torch.ones(x.shape[0], 1)
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
        p.append(self.output.params())
        return p


def optimizer_grid_search(train_loader, validation_loader, learning_rates, momentum, in_dim, out_dim, hidden_dim,
                          std, num_epochs=75, verbose=False):
    loss_heat_map = [[0 for i in learning_rates] for j in momentum]
    acc_heat_map = [[0 for i in learning_rates] for j in momentum]
    best_loss = 10000000
    for i, lr in enumerate(learning_rates):
        for j, m in enumerate(momentum):
            net = Net(in_dim, out_dim, hidden_dim, std=std)
            optimizer = torch.optim.SGD(net.params(), lr=lr, momentum=m)
            train(net, train_loader, optimizer, num_epochs=num_epochs)
            validation_loss, acc = run(net, validation_loader)
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_params = {'lr': lr, 'momentum': m}
            loss_heat_map[j][i] = validation_loss
            acc_heat_map[j][i] = acc
            if verbose:
                print('For parameters: lr- {}, momentum- {}, got loss: {}, acc- {}'.format(lr, m, validation_loss, acc))
    return best_params, best_loss, loss_heat_map, acc_heat_map

def adam_grid_search(train_loader, validation_loader, learning_rates, betas, in_dim, out_dim, hidden_dim,
                     std, num_epochs=75, verbose=False):
    loss_heat_map = [[0 for i in learning_rates] for j in betas]
    acc_heat_map = [[0 for i in learning_rates] for j in betas]
    best_loss = 10000000
    for i, lr in enumerate(learning_rates):
        for j, b in enumerate(betas):
            net = Net(in_dim, out_dim, hidden_dim, std=std)
            optimizer = torch.optim.Adam(net.params(), lr=lr, betas=b)
            train(net, train_loader, optimizer, num_epochs=num_epochs)
            validation_loss, acc = run(net, validation_loader)
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_params = {'lr': lr, 'betas': b}
            loss_heat_map[j][i] = validation_loss
            acc_heat_map[j][i] = acc
            if verbose:
                print('For parameters: lr- {}, b- {}, got loss: {}, acc- {}'.format(lr, b, validation_loss, acc))
    return best_params, best_loss, loss_heat_map, acc_heat_map


def weights_grid_search(train_loader, validation_loader, deviations, in_dim, out_dim, hidden_dim,
                        lr, m, num_epochs=75, verbose=False):
    heat_map = [0 for j in deviations]
    best_loss = 10000000
    for i, std in enumerate(deviations):
        net = Net(in_dim, out_dim, hidden_dim, std=std)
        optimizer = torch.optim.SGD(net.params(), lr=lr, momentum=m)
        train(net, train_loader, optimizer, num_epochs=num_epochs)
        validation_loss, acc = run(net, validation_loader)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_param = std
        heat_map[i] = validation_loss
        if verbose:
            print('For parameter: std- {} got loss: {}, acc: {}'.format(std, validation_loss, acc))
    return best_param, best_loss, heat_map


def adam_weights_grid_search(train_loader, validation_loader, deviations, in_dim, out_dim, hidden_dim,
                        lr, betas, num_epochs=75, verbose=False):
    heat_map = [0 for j in deviations]
    best_loss = 10000000
    for i, std in enumerate(deviations):
        net = Net(in_dim, out_dim, hidden_dim, std=std)
        optimizer = torch.optim.Adam(net.params(), lr=lr, betas=betas)
        train(net, train_loader, optimizer, num_epochs=num_epochs)
        validation_loss, acc = run(net, validation_loader)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_param = std
        heat_map[i] = validation_loss
        if verbose:
            print('For parameter: std- {} got loss: {}, acc: {}'.format(std, validation_loss, acc))
    return best_param, best_loss, heat_map

def train(net, loader, optimizer, test_set=None, verbose=False, num_epochs=75):
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    for epoch in range(num_epochs):
        train_loss = 0
        train_acc = 0
        for i, (batch, target) in enumerate(loader):
            batch = batch.cuda() if torch.cuda.is_available() else batch
            target = target.cuda() if torch.cuda.is_available() else target
            optimizer.zero_grad()
            output = net.forward(batch)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(output, target)
            loss.backward()
            optimizer.step()
            train_acc += (output.argmax(dim=1) == target.argmax(dim=1)).sum().item()
            train_loss += loss * batch.shape[0]  # avg. weighted per sample, as not all batches are equal
        # log this epoch's results
        train_loss /= len(loader.dataset)
        train_losses.append(train_loss)
        train_acc /= len(loader.dataset)
        train_accs.append(train_acc)
        if test_set is not None:
            test_loss, test_acc = run(net, test_set)
            test_losses.append(test_loss)
            test_accs.append(test_acc)
            if verbose:
                print('Epoch: {}, loss: {}, acc: {}, test loss: {}, test acc: {}'.format(epoch, train_loss, train_acc, test_loss, test_acc))
        elif verbose:
            print('Epoch: {}, loss: {}, acc: {}, test loss: {}, test acc: {}'.format(epoch, train_loss, train_acc, test_loss, test_acc))
    return train_losses, train_accs, test_losses, test_accs


def run(net, loader):
    acc = 0
    loss = 0
    for i, (batch, target) in enumerate(loader):
        batch = batch.cuda() if torch.cuda.is_available() else batch
        target= target.cuda() if torch.cuda.is_available() else target
        output = net.forward(batch)
        loss += torch.nn.functional.binary_cross_entropy_with_logits(output, target).item() * batch.shape[0]
        acc += (output.argmax(dim=1)==target.argmax(dim=1)).sum().item()
    return (loss / len(loader.dataset)), (acc / len(loader.dataset))


# Pretty-prints grid search matrix (taken off S.O.)
def show_heat_map(table, lr, m, xlabel='', ylabel='', invert=False):
    lr = [round(num,4) for num in lr]
    if type(m[0]) is tuple:
        m = [tuple([round(x,4) for x in num]) for num in m]
    else:
        m = [round(num,4) for num in m]
    #table = [[round(num, 4) for num in row] for row in table]
    df_table = pd.DataFrame(table, m, lr)
    plt.figure(figsize=(15,10))
    sn.set(font_scale=1.4)  # for label size
    if invert:
        sn.heatmap(df_table, annot=True, annot_kws={"size": 12}, cmap=sn.cm.rocket_r, fmt='.3g')  # font size
    else:
        sn.heatmap(df_table, annot=True, annot_kws={"size": 12}, fmt='.3g')  # font size
    plt.show()


def plot_train_test(train_results, test_results, epochs, label_name, cat1 = 'train ', cat2 = 'test '):
    ticks = [i for i, _ in enumerate(train_results)]
    plt.figure(figsize=(15, 10))
    plt.plot(ticks, train_results, '-b', label=(cat1 + label_name))
    plt.plot(ticks, test_results, '-r', label= (cat2 + label_name))
    plt.legend()
    plt.title('Test and train ' + label_name + ' per epoch')
    plt.show()
