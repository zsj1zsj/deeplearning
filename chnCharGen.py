import torch
import torch.nn as nn


#read file in to $content and remove empty line, remove space
qts ='./data/qts.txt'
file =open(qts,'r',encoding='utf-8')
content = [line.strip() for line in file.readlines() if line.strip()]

vocabset =set()
for line in content:
    vocabset.update(set(line))

vocab = list(vocabset)
vocab_size = len(vocab)

char2ix = {c:i for i,c in enumerate(vocab)}
ix2char = {i:c for i,c in enumerate(vocab)}



def DataLoader(content,size):
    return content[0:size],content[size+1:size+1+size]

def data2sensor(data):
    return [char2ix[c] for c in list(data)]

class MyRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyRNN, self).__init__()
        self.rnn = nn.RNN(input_size =input_size, hidden_size = hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, input):
        output,hd = self.rnn(input)
        output = self.fc(output)
        output = torch.sum(output,dim=0)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hd

hidden_size = 1000
rnn =MyRNN(vocab_size, hidden_size, vocab_size)

input = data2sensor(train_data[2][0:len(train_data[2])-1])
target = data2sensor(train_data[2][1:len(train_data[2])])

criterion = nn.NLLLoss()

learning_rate = 0.0005

import time
import math

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


n_iters = 10000
print_every = 10
plot_every = 500
all_losses = []
total_loss = 0 # Reset every plot_every iters


def train(input_line_tensor, target_line_tensor):
    rnn.zero_grad()

    loss = 0

    for i in range(len(input_line_tensor)):
        output, hidden = rnn(torch.tensor(torch.eye(vocab_size)[input_line_tensor[i]]).view(1,1,-1))
        #print(output.view(1,-1).shape)
        #print(f'target_line_tensor[i]:{ torch.tensor([target_line_tensor[i]]) }')
        l = criterion(output.view(1,-1),torch.tensor([target_line_tensor[i]]))
        
        loss+= l
                      
    loss.backward()

    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item()

start = time.time()

train_data, test_data = DataLoader(content,10000)

def train2it(train_data):
    return train_data[0:len(train_data)-1], train_data[iter][1:len(train_data)]

for iter in range(1, n_iters + 1):
    input,target = train2it(train_data[iter])
    output, loss = train(input,target)
    total_loss += loss

    print(iter)
    if iter % print_every == 0:
        print('%s (%d %d%%) %.4f' % (timeSince(start), iter, iter / n_iters * 100, loss))

    if iter % plot_every == 0:
        all_losses.append(total_loss / plot_every)
        total_loss = 0
