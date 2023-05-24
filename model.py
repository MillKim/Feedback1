import torch
import torch.nn as nn

class cfg():
    """
    cfg.py 대신에 class로 지정
    """
    def __init__(self, vs, emb = 16, hidden = 32, nh = 4, device = "cpu"):
        self.emb = emb
        self.hidden = hidden
        self.vs = vs
        self.nh = nh
        self.device = device

class RNNModel(nn.Module):
    '''
    모델 구조
    input -> embedding -> rnn -> classifier

    cfg를 이용하여 각 layer의 파라미터를 바꿀 수 있다.
    '''
    def __init__(self, out_dim, cfg):
        super(RNNModel,self).__init__()
        self.out_dim = out_dim
        self.in_dim = cfg.vs
        self.emb_dim = cfg.emb
        self.hidden_dim = cfg.hidden
        self.emb_layer = nn.Embedding(num_embeddings = self.in_dim, embedding_dim = self.emb_dim, padding_idx = 0)
        self.rnn = nn.RNN(input_size = self.emb_dim, hidden_size = self.hidden_dim, batch_first = True)
        self.classifier = nn.Linear(self.hidden_dim, self.out_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        input x : batch x time step 
        embedded : batch x time step x emb_dim
        output : batch x time step x hidden_dim
        return : batch x out_dim
        """
        embedded = self.emb_layer(x)
        output, _ = self.rnn(embedded)
        #가장 마지막 time step만 사용해주기 때문에 [:,-1,:]로 slicing 한다
        return self.sigmoid(self.classifier(output[:,-1,:]))
    

class LSTMModel(nn.Module):
    '''
    모델 구조
    input -> embedding -> lstm -> classifier

    cfg를 이용하여 각 layer의 파라미터를 바꿀 수 있다.
    '''
    def __init__(self, out_dim, cfg):
        super(LSTMModel,self).__init__()
        self.out_dim = out_dim
        self.in_dim = cfg.vs
        self.emb_dim = cfg.emb
        self.hidden_dim = cfg.hidden
        self.device = cfg.device

        self.emb_layer = nn.Embedding(num_embeddings = self.in_dim, embedding_dim = self.emb_dim, padding_idx = 0)
        self.lstm = nn.LSTM(input_size = self.emb_dim, hidden_size = self.hidden_dim, batch_first = True)
        self.classifier = nn.Linear(self.hidden_dim, self.out_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        input x : batch x time step 
        embedded : batch x time step x emb_dim
        output : batch x time step x hidden_dim
        return : batch x out_dim
        """
        h0 = torch.zeros(1, x.size(0), self.hidden_dim).to(self.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_dim).to(self.device)
        embedded = self.emb_layer(x)
        output, _ = self.lstm(embedded, (h0, c0))
        #가장 마지막 time step만 사용해주기 때문에 [:,-1,:]로 slicing 한다
        return self.sigmoid(self.classifier(output[:,-1,:]))

class ATTModel(nn.Module):
    '''
    모델 구조
    input -> embedding -> rnn -> classifier

    cfg를 이용하여 각 layer의 파라미터를 바꿀 수 있다.
    '''
    def __init__(self, out_dim, cfg):
        super(ATTModel,self).__init__()
        self.out_dim = out_dim
        self.in_dim = cfg.vs
        self.emb_dim = cfg.emb
        self.nh = cfg.nh
        self.device = cfg.device
        self.emb_layer = nn.Embedding(num_embeddings = self.in_dim, embedding_dim = self.emb_dim, padding_idx = 0)
        self.att = nn.MultiheadAttention(embed_dim = self.emb_dim, num_heads = self.nh, batch_first = True)
        self.classifier = nn.Linear(self.emb_dim, self.out_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        input x : batch x time step 
        embedded : batch x time step x emb_dim
        output : batch x time step x hidden_dim
        return : batch x out_dim
        """
        pad = self.padding_tensor(x).to(self.device)
        embedded = self.emb_layer(x)
        output, _ = self.att(query = embedded, key = embedded, value = embedded, key_padding_mask = pad)
        #가장 마지막 time step만 사용해주기 때문에 [:,-1,:]로 slicing 한다
        return self.sigmoid(self.classifier(output[:,-1,:]))

    def padding_tensor(self, x): #padding 된 값을 true로 리턴하는 함수
        zero_tensor = torch.zeros_like(x)  # 동일한 크기의 0으로 채워진 텐서 생성
        padding = torch.eq(x, zero_tensor)  # 입력 텐서와 0으로 채워진 텐서 비교
        return padding 
