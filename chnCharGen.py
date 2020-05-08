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

train_data, test_data = DataLoader(content,100)
print(len(train_data),len(test_data))

def data2sensor(data):
    return [char2ix[c] for c in list(train_data[2])]
    


# the legnth of the data is not fixed, so there's no batch in the this class, all batch_size will be set to 1.
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

input = torch.randn(1,1,vocab_size)
rnn(input)