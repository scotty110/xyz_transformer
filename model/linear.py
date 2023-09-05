from torch import nn

class FeedForward(nn.Module):
    def __init__(self, d_model, inter_size=1024):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, inter_size)
        self.linear_2 = nn.Linear(inter_size, d_model)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.9)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x

