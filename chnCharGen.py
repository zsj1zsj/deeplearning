import torch
import torch.nn as nn

content =open(qts,'r',encoding='utf-8').readlines()


# the legnth of the data is not fixed, so there's no batch in the this class, all batch_size will be set to 1.
class MyRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        self.rnn = nn.RNN(input_size =input_size, hidden_size = input_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Log_Softmax(dim=1)
    
    def forward(self, input):
        output,hd = self.rnn(input)
        output = self.fc(output)
        output = torch.sum(output,dim=0)
        output = softmax(output)
        return output, hd
        
