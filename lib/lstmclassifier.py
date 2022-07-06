import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMClassifier(nn.Module):

    def __init__(self):   
        super(LSTMClassifier, self).__init__()
        
        embedding_dim = 312
        hidden_dim = 100
        
        self.lstm = nn.LSTM(embedding_dim, 
                            hidden_dim, 
                            batch_first=True, 
                            bidirectional=True)  
        
        self.linear = nn.Linear(hidden_dim*2, 2)

    def forward(self, x):        
        x = self.lstm(x)[1][0]
        x = x.permute(1, 0, -1)
        x = torch.cat((torch.chunk(x, 2, dim=1)[0], 
                       torch.chunk(x, 2, dim=1)[1]), dim=2)
        x = x.squeeze(dim=1)
        x = self.linear(x)
        scores = F.log_softmax(x, dim=1)

        return scores

