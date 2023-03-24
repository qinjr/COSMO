import numpy as np
import torch
import torch.nn.functional as F

class FeaturesEmbedding(torch.nn.Module):
    def __init__(self, vocabulary_size, embed_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocabulary_size, embed_dim)
#         torch.nn.init.xavier_uniform_(self.embedding.weight.data)
        # maxval = np.sqrt(6. / np.sum(embed_dim))
        # minval = -maxval
        # torch.nn.init.uniform_(self.embedding.weight, minval, maxval)
    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        return self.embedding(x)


class Linear(torch.nn.Module):
    def __init__(self, vocabulary_size, output_dim=1):
        super().__init__()
        self.fc = torch.nn.Embedding(vocabulary_size, output_dim)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        return torch.sum(self.fc(x), dim=1) + self.bias


class FactorizationLayer(torch.nn.Module):
    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix #[B, 1]

class FieldAwareFactorizationLayer(torch.nn.Module):

    def __init__(self, vocabulary_size, num_fields, embed_dim):
        super().__init__()
        self.num_fields = num_fields   
        self.embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(vocabulary_size, embed_dim) for _ in range(self.num_fields)
        ]) 
        for embedding in self.embeddings:
            torch.nn.init.xavier_uniform_(embedding.weight.data)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        xs = [self.embeddings[i](x) for i in range(self.num_fields)]  
        ix = list()
        for i in range(self.num_fields - 1):
            for j in range(i + 1, self.num_fields):
                ix.append(xs[j][:, i] * xs[i][:, j])
        ix = torch.stack(ix, dim=1)
        return torch.sum(torch.sum(ix,dim=1),dim=1, keepdim=True)

class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout, output_layer=True, act = 'relu'):
        super().__init__()
        layers = list()
        for hidden_dim in hidden_dims:
            layers.append(torch.nn.Dropout(p=dropout))
            layers.append(torch.nn.Linear(input_dim, hidden_dim))
            layers.append(torch.nn.BatchNorm1d(hidden_dim))
            if act == 'relu':
                layers.append(torch.nn.ReLU())
            elif act == 'tanh':
                layers.append(torch.nn.Tanh())
            input_dim = hidden_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, input_dim)``
        """
        return self.mlp(x)


class InnerProductNetwork(torch.nn.Module):
    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        num_fields = x.shape[1]
        row, col = list(), list()
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                row.append(i), col.append(j)
        # return size: (batch_size, num_fields*(num_fields-1)/2)
        return torch.sum(x[:, row] * x[:, col], dim=2)

class CrossNet(torch.nn.Module):
    def __init__(self, input_dim, num_layers):
        super(CrossNet, self).__init__()
        self.num_layers = num_layers
        self.cross_net = torch.nn.ModuleList(CrossInteractionLayer(input_dim)
                                       for _ in range(self.num_layers))

    def forward(self, X_0):
        X_i = X_0 # b x dim
        for i in range(self.num_layers):
            X_i = X_i + self.cross_net[i](X_0, X_i)
        return X_i


class CrossInteractionLayer(torch.nn.Module):
    def __init__(self, input_dim):
        super(CrossInteractionLayer, self).__init__()
        self.weight = torch.nn.Linear(input_dim, 1, bias=False)
        self.bias = torch.nn.Parameter(torch.zeros(input_dim))

    def forward(self, X_0, X_i):
        interaction_out = self.weight(X_i) * X_0 + self.bias
        return interaction_out


class HistAtt(torch.nn.Module):
    def __init__(self, q_dim):
        super().__init__()
        self.null_attention = -2 ** 10
        input_dim = 4 * q_dim # [q, k, q * k, q - k]
        layers = list()
        for hidden_dim in [200, 80]:
            layers.append(torch.nn.Linear(input_dim, hidden_dim))
            layers.append(torch.nn.Dropout(p=0.2))
            layers.append(torch.nn.ReLU())
            input_dim = hidden_dim
        layers.append(torch.nn.Linear(input_dim, 1))
        self.atten_net = torch.nn.Sequential(*layers)
    
    def forward(self, x_item, user_hist, hist_len):
        _, len, dim = user_hist.shape # batch_size , padded_length, item_num_field*embed_dim
        x_item_tile = torch.tile(x_item.reshape([-1, 1, dim]), [1, len, 1])
        attention_inp = torch.cat((x_item_tile, user_hist, x_item_tile * user_hist, x_item_tile - user_hist), dim=2)
        score = self.atten_net(attention_inp)
        mask = torch.arange(user_hist.shape[1])[None, :].to(hist_len.device) < hist_len[:, None]
        score[~mask] = self.null_attention

        atten_score = torch.nn.Softmax(dim = 1)(score)
        user_hist_rep = torch.sum(user_hist * atten_score, dim=1)

        return user_hist_rep, score.squeeze()

class HistAtt_S(torch.nn.Module):
    def __init__(self, q_dim):
        super().__init__()
        self.null_attention = -2 ** 22
        input_dim = q_dim
        layers = list()
        for hidden_dim in [200, 80]:
            layers.append(torch.nn.Linear(input_dim, hidden_dim))
            layers.append(torch.nn.Dropout(p=0.2))
            layers.append(torch.nn.ReLU())
            input_dim = hidden_dim
        layers.append(torch.nn.Linear(input_dim, 1))
        self.atten_net = torch.nn.Sequential(*layers)
    
    def forward(self, x_item, user_hist, hist_len):
        _, len, dim = user_hist.shape
        x_item_tile = torch.tile(x_item.reshape([-1, 1, dim]), [1, len, 1])
#       attention_inp = torch.cat((x_item_tile, user_hist, x_item_tile * user_hist, x_item_tile - user_hist), dim=2)
        score = torch.sum(x_item_tile * user_hist, dim=2, keepdim=True)
        mask = torch.arange(user_hist.shape[1])[None, :].to(hist_len.device) < hist_len[:, None]
        score[~mask] = self.null_attention

        atten_score = torch.nn.Softmax(dim = 1)(score)
        user_hist_rep = torch.sum(user_hist * atten_score, dim=1)

        return user_hist_rep, score.squeeze()

class CoAtt(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.null_attention = -2 ** 22
        layers = list()
        input_dim = 3*dim
        for hidden_dim in [200, 80]:
            layers.append(torch.nn.Linear(input_dim, hidden_dim))
            layers.append(torch.nn.Dropout(p=0.2))
            layers.append(torch.nn.ReLU())
            input_dim = hidden_dim
        layers.append(torch.nn.Linear(input_dim, 1))
        self.atten_net = torch.nn.Sequential(*layers)
        self.fc1 = torch.nn.Linear(2*dim,dim)
        
    def forward(self, item_emb,x_session,session_len, user_hist, hist_len):

        batch_size,s_len,dim = x_session.shape
        h_len = user_hist.shape[1]

        user_hist_tile = torch.tile(user_hist.reshape([-1,1,h_len,dim]),[1,s_len,1,1])
        x_session_tile = torch.tile(x_session.reshape([-1,s_len,1,dim]),[1,1,h_len,1])
        item_emb_tile = torch.tile(item_emb.reshape([-1,1,1,dim]),[1,s_len,h_len,1])

        query = self.fc1(torch.cat([item_emb_tile,x_session_tile],dim=3))

        att_score = torch.sum(user_hist_tile*query,dim=3)

        # inp = torch.cat([x_session_tile,user_hist_tile,item_emb_tile],dim=3)

        # att_score = self.atten_net(inp).squeeze()

        mask_session = torch.arange(x_session.shape[1])[None, :].to(hist_len.device) < session_len[:, None]
        mask_session = torch.tile(mask_session.reshape(-1,s_len,1),[1,1,h_len])
        mask_hist =torch.arange(user_hist.shape[1])[None, :].to(hist_len.device) < hist_len[:, None]
        mask_hist = torch.tile(mask_hist.reshape(-1,1,h_len),[1,s_len,1])

        att_score[~mask_session] = self.null_attention
        att_score[~mask_hist] = self.null_attention

        #先softmax再mean
        # att_score = (torch.nn.Softmax(dim=1)(att_score.reshape(-1,s_len*h_len))).reshape(-1,s_len,h_len)
        # att_score = torch.sum(att_score,dim=1)
        
        #先max再softmax
        score = torch.max(att_score.squeeze(),dim=1)[0]
        att_score = torch.nn.Softmax(dim=1)(score)

        user_hist_rep = torch.sum((user_hist * (att_score.unsqueeze(dim=2))), dim=1)
        
        return user_hist_rep, score