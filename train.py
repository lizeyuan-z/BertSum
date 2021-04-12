import torch
from torch.optim import Adam, SGD
import torch.nn as nn
from model import BertSumTransformer, BertSumClassifier


def train(data_setup, pretrain_model, device, max_length, learning_rate, trained_model):

    # load data
    data = []
    for i in range(data_setup[1]):
        data += torch.load(data_setup[0].format(i))

    # load model
    model = BertSumClassifier(pretrain_model, device, max_length)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate, eps=1e-8)
    loss_fn = nn.CrossEntropyLoss()
    loss_fn.to(device)
    optimizer.zero_grad()

    # train model
    for i in range(len(data)):
        out = model(data[i]['src'], data[i]['clss']).to(device)
        loss = loss_fn(out, torch.tensor(data[i]['labels'], dtype=torch.long)).to(device)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print('>>>batch:{0}     loss:{1}'.format(i, loss))
    torch.save(model.state_dict(), 'trained_model' + '/{0}_model'.format(trained_model))


if __name__ == '__main__':
    data_setup = ['data/cnndm.train.{0}.bert.pt', 143]
    pretrain_model = 'pertrain_model'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    max_length = 500
    learning_rate = 2e-5
    trained_model = 'BertSum'
    train(data_setup, pretrain_model, device, max_length, learning_rate, trained_model)
