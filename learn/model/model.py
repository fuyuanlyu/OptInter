import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import drop_path
import numpy as np
import sys

class LR(nn.Module):
    def __init__(self, cont_field, cate_field, cate_cont_feature, 
                device=torch.device('cpu'), lamb=0.):
        super(LR, self).__init__()
        self.cont_field = cont_field
        self.cate_field = cate_field
        self.cate_cont_feature = cate_cont_feature
        self.device = device
        self.lamb = lamb

        # Create embedding table
        self.cate_embeddings_table = \
            nn.Embedding(self.cate_cont_feature, 1)

        # Initialize
        for name, tensor in self.cate_embeddings_table.named_parameters():
            if 'weight' in name:
                a = np.square(3/(self.cate_field * 1))
                nn.init.uniform_(tensor, -a, a)

    def forward(self, conts, cates, combs):
        # Assert the batch sizes are the same
        batch_size = conts.size()[0]
        assert batch_size == cates.size()[0]

        # Get continuous, categorical and free combinad embeddings
        cont_embedding = torch.IntTensor(np.arange(self.cont_field))\
                .expand_as(conts).to(self.device)
        cont_embedding = self.cate_embeddings_table(cont_embedding)
        conts = conts.unsqueeze(2)
        cont_embedding = torch.mul(cont_embedding, conts)
        cate_embedding = self.cate_embeddings_table(cates)
        
        # Compute and reshape combined embeddings
        X = torch.cat((cont_embedding, cate_embedding), 1).reshape(batch_size, -1)

        # LR part
        logit = torch.sigmoid(torch.sum(X, dim=1, keepdim=True))

        return logit

    def l2_penalty(self, conts, cates, combs):
        # Assert the batch sizes are the same
        batch_size = conts.size()[0] 
        assert batch_size == (cates.size()[0])
        conts = conts.reshape(batch_size, -1)
        cates = cates.reshape(batch_size, -1)

        # Get continuous, categorical and free combinad embeddings
        cont_embedding = torch.IntTensor(np.arange(self.cont_field))\
                .expand_as(conts).to(self.device)
        cont_embedding = self.cate_embeddings_table(cont_embedding)
        conts = conts.unsqueeze(2)
        cont_embedding = torch.mul(cont_embedding, conts)
        cate_embedding = self.cate_embeddings_table(cates)

        # Compute and reshape combined embeddings
        X = torch.cat((cont_embedding, cate_embedding), 1).reshape(batch_size, -1)

        # Calculate L2
        L2 = torch.pow(X, 2) * self.lamb
        L2 = L2.sum()

        return L2


class FM(nn.Module):
    def __init__(self, cont_field, cate_field, cate_cont_feature, 
                orig_embedding_dim=40, hidden_dims=[100,100],
                device=torch.device('cpu'), lamb=0.):
        super(FM, self).__init__()
        self.cont_field = cont_field
        self.cate_field = cate_field
        self.cate_cont_feature = cate_cont_feature
        self.orig_embedding_dim = orig_embedding_dim
        self.device = device
        self.lamb = lamb

        # Create embedding table
        self.cate_embeddings_table = \
            nn.Embedding(self.cate_cont_feature, self.orig_embedding_dim)

        for name, tensor in self.cate_embeddings_table.named_parameters():
            if 'weight' in name:
                a = np.square(3/(self.cate_field * self.orig_embedding_dim))
                nn.init.uniform_(tensor, -a, a)

    def forward(self, conts, cates, combs):
        # Assert the batch sizes are the same
        batch_size = conts.size()[0]
        assert batch_size == cates.size()[0]

        # Get continuous, categorical and free combinad embeddings
        cont_embedding = torch.IntTensor(np.arange(self.cont_field))\
                .expand_as(conts).to(self.device)
        cont_embedding = self.cate_embeddings_table(cont_embedding)
        conts = conts.unsqueeze(2)
        cont_embedding = torch.mul(cont_embedding, conts)
        cate_embedding = self.cate_embeddings_table(cates)

        # FM part
        cont_cate_embedding = torch.cat((cont_embedding, cate_embedding), 1)
        square_of_sum = torch.sum(cont_cate_embedding, dim=1) ** 2
        sum_of_square = torch.sum(cont_cate_embedding ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        ix = 0.5 * ix
        X_FM = torch.sum(ix, dim=1, keepdim=True)

        logit = torch.sigmoid(X_FM)

        return logit

    def l2_penalty(self, conts, cates, combs):
        # Assert the batch sizes are the same
        batch_size = conts.size()[0] 
        assert batch_size == (cates.size()[0])
        conts = conts.reshape(batch_size, -1)
        cates = cates.reshape(batch_size, -1)

        # Get continuous, categorical and free combinad embeddings
        cont_embedding = torch.IntTensor(np.arange(self.cont_field))\
                .expand_as(conts).to(self.device)
        cont_embedding = self.cate_embeddings_table(cont_embedding)
        conts = conts.unsqueeze(2)
        cont_embedding = torch.mul(cont_embedding, conts)
        cate_embedding = self.cate_embeddings_table(cates)

        # Compute and reshape combined embeddings
        X = torch.cat((cont_embedding, cate_embedding), 1).reshape(batch_size, -1)

        # Calculate L2
        L2 = torch.pow(X, 2) * self.lamb
        L2 = L2.sum()

        return L2


class Poly2(nn.Module):
    def __init__(self, cont_field, cate_field, comb_field, cate_cont_feature, comb_feature,
                device=torch.device('cpu'), lamb=0.):
        super(Poly2, self).__init__()
        self.cont_field = cont_field
        self.cate_field = cate_field
        self.comb_field = comb_field
        self.cate_cont_feature = cate_cont_feature
        self.comb_feature = comb_feature
        self.device = device
        self.lamb = lamb

        # Create embedding table
        self.cate_embeddings_table = \
            nn.Embedding(self.cate_cont_feature, 1)
        self.comb_embeddings_table = \
            nn.Embedding(self.comb_feature, 1)

        # Initialize
        for name, tensor in self.cate_embeddings_table.named_parameters():
            if 'weight' in name:
                a = np.square(3/(self.cate_field * 1))
                nn.init.uniform_(tensor, -a, a)
        for name, tensor in self.comb_embeddings_table.named_parameters():
            if 'weight' in name:
                a = np.square(3/(self.comb_field * 1))
                nn.init.uniform_(tensor, -a, a)

    def forward(self, conts, cates, combs):
        # Assert the batch sizes are the same
        batch_size = conts.size()[0]
        assert batch_size == cates.size()[0]
        assert batch_size == combs.size()[0]

        # Get continuous, categorical and free combined embeddings
        cont_embedding = torch.IntTensor(np.arange(self.cont_field))\
                .expand_as(conts).to(self.device)
        cont_embedding = self.cate_embeddings_table(cont_embedding)
        conts = conts.unsqueeze(2)
        cont_embedding = torch.mul(cont_embedding, conts)
        cate_embedding = self.cate_embeddings_table(cates)

        # Reshape all embeddings
        # Compute original features
        cont_cate_embedding = torch.cat((cont_embedding, cate_embedding), 1) \
                .type(torch.FloatTensor).to(self.device)
        comb_embedding = self.comb_embeddings_table(combs)

        # Compute final X as model input
        X = torch.cat((cont_cate_embedding.reshape(batch_size, -1), 
            comb_embedding.reshape(batch_size, -1)), 1)\
            .type(torch.FloatTensor).to(self.device)

        # LR part
        logit = torch.sigmoid(torch.sum(X, dim=1, keepdim=True))

        return logit

    def l2_penalty(self, conts, cates, combs):
        # Assert the batch sizes are the same
        batch_size = conts.size()[0] 
        assert batch_size == (cates.size()[0])
        conts = conts.reshape(batch_size, -1)
        cates = cates.reshape(batch_size, -1)

        # Get continuous, categorical and free combinad embeddings
        cont_embedding = torch.IntTensor(np.arange(self.cont_field))\
                .expand_as(conts).to(self.device)
        cont_embedding = self.cate_embeddings_table(cont_embedding)
        conts = conts.unsqueeze(2)
        cont_embedding = torch.mul(cont_embedding, conts)
        cate_embedding = self.cate_embeddings_table(cates)
        comb_embedding = self.comb_embeddings_table(combs)

        # Compute and reshape combined embeddings
        X = torch.cat((cont_embedding, cate_embedding, comb_embedding), 1).reshape(batch_size, -1)

        # Calculate L2
        L2 = torch.pow(X, 2) * self.lamb
        L2 = L2.sum()

        return L2


class FNN(nn.Module):
    def __init__(self, cont_field, cate_field, cate_cont_feature,
                orig_embedding_dim=40, hidden_dims=[100,100],
                device=torch.device('cpu'), lamb=0.):
        super(FNN, self).__init__()
        self.cont_field = cont_field
        self.cate_field = cate_field
        self.cate_cont_feature = cate_cont_feature
        self.orig_embedding_dim = orig_embedding_dim
        self.device = device
        self.lamb = lamb

        # Create embedding table
        self.cate_embeddings_table = \
            nn.Embedding(self.cate_cont_feature, self.orig_embedding_dim)

        # Create layers
        self.fc_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        
        first_layer_neurons = self.orig_embedding_dim * \
                (self.cate_field + self.cont_field)

        self.fc_layers.append(nn.Linear(first_layer_neurons, hidden_dims[0]))
        for _, (in_size, out_size) in enumerate(zip(hidden_dims[:-1], hidden_dims[1:])):
            self.fc_layers.append(nn.Linear(in_size, out_size))

        for _, size in enumerate(hidden_dims):
            self.norm_layers.append(nn.LayerNorm(size))
        
        self.output_layer = nn.Linear(hidden_dims[-1], 1)

        for name, tensor in self.fc_layers.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(tensor, gain=1)
        for name, tensor in self.output_layer.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(tensor, gain=1)
        for name, tensor in self.cate_embeddings_table.named_parameters():
            if 'weight' in name:
                a = np.square(3/(self.cate_field * self.orig_embedding_dim))
                nn.init.uniform_(tensor, -a, a)

    def forward(self, conts, cates, combs):
        # Assert the batch sizes are the same
        batch_size = conts.size()[0] 
        assert batch_size == (cates.size()[0])
        conts = conts.reshape(batch_size, -1)
        cates = cates.reshape(batch_size, -1)

        # Get continuous, categorical and free combinad embeddings
        cont_embedding = torch.IntTensor(np.arange(self.cont_field))\
                .expand_as(conts).to(self.device)
        cont_embedding = self.cate_embeddings_table(cont_embedding)
        conts = conts.unsqueeze(2)
        cont_embedding = torch.mul(cont_embedding, conts)
        cate_embedding = self.cate_embeddings_table(cates)

        # Compute and reshape combined embeddings
        X = torch.cat((cont_embedding, cate_embedding), 1).reshape(batch_size, -1)

        # Pass to FC layers
        for idx in range(len(self.fc_layers)):
            X = self.fc_layers[idx](X)
            X = self.norm_layers[idx](X)
            X = F.relu(X)
        logit = self.output_layer(X)
        logit = torch.sigmoid(logit)

        return logit

    def l2_penalty(self, conts, cates, combs):
        # Assert the batch sizes are the same
        batch_size = conts.size()[0] 
        assert batch_size == (cates.size()[0])
        conts = conts.reshape(batch_size, -1)
        cates = cates.reshape(batch_size, -1)

        # Get continuous, categorical and free combinad embeddings
        cont_embedding = torch.IntTensor(np.arange(self.cont_field))\
                .expand_as(conts).to(self.device)
        cont_embedding = self.cate_embeddings_table(cont_embedding)
        conts = conts.unsqueeze(2)
        cont_embedding = torch.mul(cont_embedding, conts)
        cate_embedding = self.cate_embeddings_table(cates)

        # Compute and reshape combined embeddings
        X = torch.cat((cont_embedding, cate_embedding), 1).reshape(batch_size, -1)

        # Calculate L2
        L2 = torch.pow(X, 2) * self.lamb
        L2 = L2.sum()

        return L2


class IPNN(nn.Module):
    def __init__(self, cont_field, cate_field, cate_cont_feature,
                orig_embedding_dim=40, hidden_dims=[100,100],
                device=torch.device('cpu'), lamb=0.):
        super(IPNN, self).__init__()
        self.cont_field = cont_field
        self.cate_field = cate_field
        self.cate_cont_feature = cate_cont_feature
        self.orig_embedding_dim = orig_embedding_dim
        self.device = device
        self.lamb = lamb

        # Compute comb_field
        self.cont_cate_field = self.cate_field + self.cont_field
        self.comb_field = int(self.cont_cate_field * (self.cont_cate_field - 1) / 2)

        # Create embedding table
        self.cate_embeddings_table = \
            nn.Embedding(self.cate_cont_feature, self.orig_embedding_dim)

        # Create layers
        self.fc_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        
        first_layer_neurons = self.orig_embedding_dim * \
                (self.cate_field + self.cont_field) + self.comb_field

        self.fc_layers.append(nn.Linear(first_layer_neurons, hidden_dims[0]))
        for _, (in_size, out_size) in enumerate(zip(hidden_dims[:-1], hidden_dims[1:])):
            self.fc_layers.append(nn.Linear(in_size, out_size))

        for _, size in enumerate(hidden_dims):
            self.norm_layers.append(nn.LayerNorm(size))
        
        self.output_layer = nn.Linear(hidden_dims[-1], 1)

        for name, tensor in self.fc_layers.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(tensor, gain=1)
        for name, tensor in self.output_layer.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(tensor, gain=1)
        for name, tensor in self.cate_embeddings_table.named_parameters():
            if 'weight' in name:
                a = np.square(3/(self.cate_field * self.orig_embedding_dim))
                nn.init.uniform_(tensor, -a, a)

        # Create indexes
        rows = []
        cols = []
        for i in range(self.cont_cate_field):
            for j in range(i+1, self.cont_cate_field):
                rows.append(i)
                cols.append(j)
        self.rows = torch.tensor(rows, device=self.device)
        self.cols = torch.tensor(cols, device=self.device)
    
    def forward(self, conts, cates, combs):
        # Assert the batch sizes are the same
        batch_size = conts.size()[0]
        assert batch_size == cates.size()[0]

        # Get continuous and categorical embeddings
        cont_embedding = torch.IntTensor(np.arange(self.cont_field))\
                .expand_as(conts).to(self.device)
        cont_embedding = self.cate_embeddings_table(cont_embedding)
        conts = conts.unsqueeze(2)
        cont_embedding = torch.mul(cont_embedding, conts)
        cate_embedding = self.cate_embeddings_table(cates)

        cont_cate_embedding = torch.cat((cont_embedding, cate_embedding), 1)

        # Compute and reshape combined embeddings
        trans = torch.transpose(cont_cate_embedding, 1, 2)
        gather_rows = torch.gather(trans, 2, self.rows.expand(batch_size, trans.shape[1], self.rows.shape[0]))
        gather_cols = torch.gather(trans, 2, self.cols.expand(batch_size, trans.shape[1], self.rows.shape[0]))
        p = torch.transpose(gather_rows, 1, 2)
        q = torch.transpose(gather_cols, 1, 2)
        comp_comb_embedding = torch.mul(p, q)
        comp_comb_embedding = torch.sum(comp_comb_embedding, 2)
        cont_embedding = cont_embedding.reshape(batch_size, -1)
        cate_embedding = cate_embedding.reshape(batch_size, -1)
        X = torch.cat((cont_embedding, cate_embedding, comp_comb_embedding), 1)
            
        # Pass to FC layers
        for idx in range(len(self.fc_layers)):
            X = self.fc_layers[idx](X)
            X = self.norm_layers[idx](X)
            X = F.relu(X)
        logit = self.output_layer(X)
        logit = torch.sigmoid(logit)

        return logit

    def l2_penalty(self, conts, cates, combs):
        # Assert the batch sizes are the same
        batch_size = conts.size()[0] 
        assert batch_size == (cates.size()[0])
        conts = conts.reshape(batch_size, -1)
        cates = cates.reshape(batch_size, -1)
        
        # Get continuous, categorical and free combinad embeddings
        cont_embedding = torch.IntTensor(np.arange(self.cont_field))\
                .expand_as(conts).to(self.device)
        cont_embedding = self.cate_embeddings_table(cont_embedding)
        conts = conts.unsqueeze(2)
        cont_embedding = torch.mul(cont_embedding, conts)
        cate_embedding = self.cate_embeddings_table(cates)

        # Compute and reshape combined embeddings
        X = torch.cat((cont_embedding, cate_embedding), 1).reshape(batch_size, -1)

        # Calculate L2
        L2 = torch.pow(X, 2) * self.lamb
        L2 = L2.sum()

        return L2


class DeepFM(nn.Module):
    def __init__(self, cont_field, cate_field, cate_cont_feature,
                orig_embedding_dim=40, hidden_dims=[100,100], 
                device=torch.device('cpu'), lamb=0.):
        super(DeepFM, self).__init__()
        self.cont_field = cont_field
        self.cate_field = cate_field
        self.cate_cont_feature = cate_cont_feature
        self.orig_embedding_dim = orig_embedding_dim
        self.device = device
        self.lamb = lamb

        # Compute comb_field
        self.cont_cate_field = self.cate_field + self.cont_field

        # Create embedding table
        self.cate_embeddings_table = \
            nn.Embedding(self.cate_cont_feature, self.orig_embedding_dim)

        # Deep part
        self.fc_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        
        first_layer_neurons = self.orig_embedding_dim * self.cont_cate_field

        self.fc_layers.append(nn.Linear(first_layer_neurons, hidden_dims[0]))
        for _, (in_size, out_size) in enumerate(zip(hidden_dims[:-1], hidden_dims[1:])):
            self.fc_layers.append(nn.Linear(in_size, out_size))

        for _, size in enumerate(hidden_dims):
            self.norm_layers.append(nn.LayerNorm(size))

        self.output_layer = nn.Linear(hidden_dims[-1], 1)

        for name, tensor in self.fc_layers.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(tensor, gain=1)
        for name, tensor in self.output_layer.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(tensor, gain=1)
        for name, tensor in self.cate_embeddings_table.named_parameters():
            if 'weight' in name:
                a = np.square(3/(self.cate_field * self.orig_embedding_dim))
                nn.init.uniform_(tensor, -a, a)
    
    def forward(self, conts, cates, combs):
        # Assert the batch sizes are the same
        batch_size = conts.size()[0]
        assert batch_size == cates.size()[0]

        # Get continuous, categorical and free combinad embeddings
        cont_embedding = torch.IntTensor(np.arange(self.cont_field))\
                .expand_as(conts).to(self.device)
        cont_embedding = self.cate_embeddings_table(cont_embedding)
        conts = conts.unsqueeze(2)
        cont_embedding = torch.mul(cont_embedding, conts)
        cate_embedding = self.cate_embeddings_table(cates)

        # FM part
        cont_cate_embedding = torch.cat((cont_embedding, cate_embedding), 1)
        square_of_sum = torch.sum(cont_cate_embedding, dim=1) ** 2
        sum_of_square = torch.sum(cont_cate_embedding ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        ix = 0.5 * ix
        X_FM = torch.sum(ix, dim=1, keepdim=True)

        # Deep part
        cont_embedding = cont_embedding.reshape(batch_size, -1)
        cate_embedding = cate_embedding.reshape(batch_size, -1)
        X_DNN = torch.cat((cont_embedding, cate_embedding), 1)
            
        for idx in range(len(self.fc_layers)):
            X_DNN = self.fc_layers[idx](X_DNN)
            X_DNN = self.norm_layers[idx](X_DNN)
            X_DNN = F.relu(X_DNN)
        X_DNN = self.output_layer(X_DNN)

        logit = torch.sigmoid(X_FM + X_DNN)

        return logit

    def l2_penalty(self, conts, cates, combs):
        # Assert the batch sizes are the same
        batch_size = conts.size()[0] 
        assert batch_size == (cates.size()[0])
        conts = conts.reshape(batch_size, -1)
        cates = cates.reshape(batch_size, -1)
        
        # Get continuous, categorical and free combinad embeddings
        cont_embedding = torch.IntTensor(np.arange(self.cont_field))\
                .expand_as(conts).to(self.device)
        cont_embedding = self.cate_embeddings_table(cont_embedding)
        conts = conts.unsqueeze(2)
        cont_embedding = torch.mul(cont_embedding, conts)
        cate_embedding = self.cate_embeddings_table(cates)

        # Compute and reshape combined embeddings
        X = torch.cat((cont_embedding, cate_embedding), 1).reshape(batch_size, -1)

        # Calculate L2
        L2 = torch.pow(X, 2) * self.lamb
        L2 = L2.sum()

        return L2


class PIN(nn.Module):
    def __init__(self, cont_field, cate_field, cate_cont_feature,
                orig_embedding_dim=40, hidden_dims=[100,100], subnet=[40,5],
                device=torch.device('cpu'), lamb=0.):
        super(PIN, self).__init__()
        self.cont_field = cont_field
        self.cate_field = cate_field
        self.cate_cont_feature = cate_cont_feature
        self.orig_embedding_dim = orig_embedding_dim
        self.device = device
        self.lamb = lamb

        # Compute comb_field
        self.cont_cate_field = self.cate_field + self.cont_field
        self.comb_field = int(self.cont_cate_field * (self.cont_cate_field - 1) / 2)

        # Create embedding table
        self.cate_embeddings_table = \
            nn.Embedding(self.cate_cont_feature, self.orig_embedding_dim)

        # Create layers
        self.fc_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        
        first_layer_neurons = self.comb_field * subnet[-1] + self.cont_cate_field * self.orig_embedding_dim

        self.fc_layers.append(nn.Linear(first_layer_neurons, hidden_dims[0]))
        for _, (in_size, out_size) in enumerate(zip(hidden_dims[:-1], hidden_dims[1:])):
            self.fc_layers.append(nn.Linear(in_size, out_size))

        for _, size in enumerate(hidden_dims):
            self.norm_layers.append(nn.LayerNorm(size))
        
        self.output_layer = nn.Linear(hidden_dims[-1], 1)

        # Create sub-net
        self.sub_norm_layers = nn.ModuleList()

        self.sub_w = []
        self.sub_b = []
        layer_input = self.orig_embedding_dim * 3
        for idx, layer_output in enumerate(subnet):
            self.sub_w.append(torch.empty(self.comb_field, layer_input, layer_output, dtype=torch.float32, device=self.device, requires_grad=True))
            self.sub_b.append(torch.empty(self.comb_field, 1, layer_output, dtype=torch.float32, device=self.device, requires_grad=True))
            layer_input = layer_output

        for _, size in enumerate(subnet):
            self.sub_norm_layers.append(nn.LayerNorm(size))

        # Initialization
        for name, tensor in self.fc_layers.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(tensor, gain=1)
        for name, tensor in self.output_layer.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(tensor, gain=1)
        for name, tensor in self.cate_embeddings_table.named_parameters():
            if 'weight' in name:
                a = np.square(3/(self.cate_field * self.orig_embedding_dim))
                nn.init.uniform_(tensor, -a, a)

        for idx in range(len(self.sub_w)):
            nn.init.xavier_uniform(self.sub_w[idx], gain=1)
            nn.init.xavier_uniform(self.sub_b[idx], gain=1)

        # Create indexes
        rows = []
        cols = []
        for i in range(self.cont_cate_field):
            for j in range(i+1, self.cont_cate_field):
                rows.append(i)
                cols.append(j)
        self.rows = torch.tensor(rows, device=self.device)
        self.cols = torch.tensor(cols, device=self.device)
    
    def forward(self, conts, cates, combs):
        # Assert the batch sizes are the same
        batch_size = conts.size()[0]
        assert batch_size == cates.size()[0]

        # Get continuous and categorical embeddings
        cont_embedding = torch.IntTensor(np.arange(self.cont_field))\
                .expand_as(conts).to(self.device)
        cont_embedding = self.cate_embeddings_table(cont_embedding)
        conts = conts.unsqueeze(2)
        cont_embedding = torch.mul(cont_embedding, conts)
        cate_embedding = self.cate_embeddings_table(cates)

        cont_cate_embedding = torch.cat((cont_embedding, cate_embedding), 1)

        # Compute and reshape combined embeddings
        trans = torch.transpose(cont_cate_embedding, 1, 2)
        gather_rows = torch.gather(trans, 2, self.rows.expand(batch_size, trans.shape[1], self.rows.shape[0]))
        gather_cols = torch.gather(trans, 2, self.cols.expand(batch_size, trans.shape[1], self.rows.shape[0]))
        p = torch.transpose(gather_rows, 1, 2)
        q = torch.transpose(gather_cols, 1, 2)
        comp_comb_embedding = torch.mul(p, q)
        z = torch.cat((p,q,comp_comb_embedding), 2)
        z = torch.transpose(z, 0, 1)
        for idx in range(len(self.sub_norm_layers)):
            z = torch.matmul(z, self.sub_w[idx])
            z = z + self.sub_b[idx] 
            z = self.sub_norm_layers[idx](z)
            z = F.relu(z)
        z = torch.transpose(z, 0, 1)
        z = z.reshape(batch_size, -1)
        cont_cate_embedding = cont_cate_embedding.reshape(batch_size, -1)
        X = torch.cat((cont_cate_embedding, z), 1)
        
        # Pass to FC layers
        for idx in range(len(self.fc_layers)):
            X = self.fc_layers[idx](X)
            X = self.norm_layers[idx](X)
            X = F.relu(X)
        logit = self.output_layer(X)
        logit = torch.sigmoid(logit)

        return logit

    def l2_penalty(self, conts, cates, combs):
        # Assert the batch sizes are the same
        batch_size = conts.size()[0] 
        assert batch_size == (cates.size()[0])
        conts = conts.reshape(batch_size, -1)
        cates = cates.reshape(batch_size, -1)
        
        # Get continuous, categorical and free combinad embeddings
        cont_embedding = torch.IntTensor(np.arange(self.cont_field))\
                .expand_as(conts).to(self.device)
        cont_embedding = self.cate_embeddings_table(cont_embedding)
        conts = conts.unsqueeze(2)
        cont_embedding = torch.mul(cont_embedding, conts)
        cate_embedding = self.cate_embeddings_table(cates)

        # Compute and reshape combined embeddings
        X = torch.cat((cont_embedding, cate_embedding), 1).reshape(batch_size, -1)

        # Calculate L2
        L2 = torch.pow(X, 2) * self.lamb
        L2 = L2.sum()

        return L2


class DNN_cart(nn.Module):
    def __init__(self, cont_field, cate_field, comb_field, cate_cont_feature, comb_feature,
                arch=0, orig_embedding_dim=40, comb_embedding_dim=10, hidden_dims=[100,100],
                device=torch.device('cpu'), alpha_mode=0, selected_pairs=None, lamb=0.):
        super(DNN_cart, self).__init__()
        self.cont_field = cont_field
        self.cate_field = cate_field
        self.comb_field = comb_field
        self.cate_cont_feature = cate_cont_feature
        self.comb_feature = comb_feature
        self.orig_embedding_dim = orig_embedding_dim
        self.comb_embedding_dim = comb_embedding_dim
        self.device = device
        self.alpha_mode = alpha_mode
        if self.alpha_mode == 0:
            self.arch = torch.from_numpy(arch).to(self.device)
        if self.alpha_mode in [0,2]:
            if selected_pairs == None:
                self.selected_pairs = []
                cont_cate_fields = self.cont_field + self.cate_field
                for i in range(cont_cate_fields):
                    for j in range(i+1, cont_cate_fields):
                        self.selected_pairs.append((i,j))
            else:
                self.selected_pairs = selected_pairs
        self.lamb = lamb
        
        # Create embedding tables
        self.cate_embeddings_table = \
            nn.Embedding(self.cate_cont_feature, self.orig_embedding_dim)
        if self.alpha_mode in [0,1]:
            self.comb_embeddings_table = \
                nn.Embedding(self.comb_feature, self.comb_embedding_dim)
        if self.alpha_mode in [0,2]:
            self.addition_embeddings_table = \
                nn.Embedding(self.cate_cont_feature, self.comb_embedding_dim)

        # Create layers
        self.fc_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        first_layer_neurons = self.orig_embedding_dim * \
                (self.cate_field + self.cont_field) + self.comb_embedding_dim * self.comb_field

        self.fc_layers.append(nn.Linear(first_layer_neurons, hidden_dims[0]))
        for _, (in_size, out_size) in enumerate(zip(hidden_dims[:-1], hidden_dims[1:])):
            self.fc_layers.append(nn.Linear(in_size, out_size))

        for _, size in enumerate(hidden_dims):
            self.norm_layers.append(nn.LayerNorm(size))

        self.output_layer = nn.Linear(hidden_dims[-1], 1)

        for name, tensor in self.fc_layers.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(tensor, gain=1)
        for name, tensor in self.output_layer.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(tensor, gain=1)
        for name, tensor in self.cate_embeddings_table.named_parameters():
            if 'weight' in name:
                a = np.square(3/(self.cate_field * self.orig_embedding_dim))
                nn.init.uniform_(tensor, -a, a)
        if hasattr(self, 'addition_embeddings_table'):
            for name, tensor in self.addition_embeddings_table.named_parameters():
                if 'weight' in name:
                    a = np.square(3/(self.cate_field * self.comb_embedding_dim))
                    nn.init.uniform_(tensor, -a, a)
        if hasattr(self, 'comb_embeddings_table'):
            for name, tensor in self.comb_embeddings_table.named_parameters():
                if 'weight' in name:
                    a = np.square(3/(self.comb_field * self.comb_embedding_dim))
                    nn.init.uniform_(tensor, -a, a)
        
    def forward(self, conts, cates, combs):
        # Assert the batch sizes are the same
        batch_size = conts.size()[0]
        assert batch_size == cates.size()[0]
        assert batch_size == combs.size()[0]

        # Get continuous, categorical and free combinad embeddings
        cont_embedding = torch.IntTensor(np.arange(self.cont_field))\
                .expand_as(conts).to(self.device)
        cont_embedding = self.cate_embeddings_table(cont_embedding)
        conts = conts.unsqueeze(2)
        cont_embedding = torch.mul(cont_embedding, conts)
        cate_embedding = self.cate_embeddings_table(cates)

        # Reshape all embeddings
        # Compute original features
        cont_cate_embedding = torch.cat((cont_embedding, cate_embedding), 1) \
                .type(torch.FloatTensor).to(self.device)

        # Expand combined embedding dimension
        if self.alpha_mode in [0,1]:
            comb_embedding = self.comb_embeddings_table(combs)

        # Compute combined embeddings
        if self.alpha_mode in [0,2]:
            addition_cate_embedding = self.addition_embeddings_table(cates)
            for index, (i,j) in enumerate(self.selected_pairs):
                embedding_i = addition_cate_embedding[:,i]
                embedding_j = addition_cate_embedding[:,j]
                if index == 0:
                    comp_comb_embedding = embedding_i.mul(embedding_j)\
                        .unsqueeze(1)
                else:
                    comp_comb_embedding = torch.cat((comp_comb_embedding, \
                        embedding_i.mul(embedding_j).unsqueeze(1)), 1)

        # Null embedding
        if self.alpha_mode in [0,3]:
            null_embedding = torch.zeros(batch_size, self.comb_field, 
                self.comb_embedding_dim, device=self.device)

        # Compute final combined embedding
        if self.alpha_mode == 0:
            final_comb_embedding = comb_embedding.mul(self.arch[:,0].unsqueeze(0).unsqueeze(2)) \
                            + comp_comb_embedding.mul(self.arch[:,1].unsqueeze(0).unsqueeze(2)) \
                            + null_embedding.mul(self.arch[:,2].unsqueeze(0).unsqueeze(2))
        elif self.alpha_mode == 1:
            final_comb_embedding = comb_embedding
        elif self.alpha_mode == 2:
            final_comb_embedding = comp_comb_embedding
        elif self.alpha_mode == 3:
            final_comb_embedding = null_embedding
        
        # Compute final X as model input
        X = torch.cat((cont_cate_embedding.reshape(batch_size, -1), 
            final_comb_embedding.reshape(batch_size, -1)), 1)\
            .type(torch.FloatTensor).to(self.device)

        # Pass to FC layers
        for idx in range(len(self.fc_layers)):
            X = self.fc_layers[idx](X)
            X = self.norm_layers[idx](X)
            X = F.relu(X)
        logit = self.output_layer(X)
        logit = torch.sigmoid(logit)

        return logit

    # def l2_penalty(self, conts, cates, combs):
    #     # Assert the batch sizes are the same
    #     batch_size = conts.size()[0]
    #     assert batch_size == cates.size()[0]
    #     assert batch_size == combs.size()[0]

    #     # Get continuous, categorical and free combinad embeddings
    #     cont_embedding = torch.IntTensor(np.arange(self.cont_field))\
    #             .expand_as(conts).to(self.device)
    #     cont_embedding = self.cate_embeddings_table(cont_embedding)
    #     conts = conts.unsqueeze(2)
    #     cont_embedding = torch.mul(cont_embedding, conts)
    #     cate_embedding = self.cate_embeddings_table(cates)

    #     # Reshape all embeddings
    #     # Compute original features
    #     cont_cate_embedding = torch.cat((cont_embedding, cate_embedding), 1) \
    #             .type(torch.FloatTensor).to(self.device)

    #     # Expand combined embedding dimension
    #     if self.alpha_mode in [0,1]:
    #         comb_embedding = self.comb_embeddings_table(combs)

    #     # Compute combined embeddings
    #     if self.alpha_mode in [0,2]:
    #         addition_cate_embedding = self.addition_embeddings_table(cates)
    #         for index, (i,j) in enumerate(self.selected_pairs):
    #             embedding_i = addition_cate_embedding[:,i]
    #             embedding_j = addition_cate_embedding[:,j]
    #             if index == 0:
    #                 comp_comb_embedding = embedding_i.mul(embedding_j)\
    #                     .unsqueeze(1)
    #             else:
    #                 comp_comb_embedding = torch.cat((comp_comb_embedding, \
    #                     embedding_i.mul(embedding_j).unsqueeze(1)), 1)

    #     # Null embedding
    #     if self.alpha_mode in [0,3]:
    #         null_embedding = torch.zeros(batch_size, self.comb_field, 
    #             self.comb_embedding_dim, device=self.device)

    #     # Compute final combined embedding
    #     if self.alpha_mode == 0:
    #         final_comb_embedding = comb_embedding.mul(self.arch[:,0].unsqueeze(0).unsqueeze(2)) \
    #                         + comp_comb_embedding.mul(self.arch[:,1].unsqueeze(0).unsqueeze(2)) \
    #                         + null_embedding.mul(self.arch[:,2].unsqueeze(0).unsqueeze(2))
    #     elif self.alpha_mode == 1:
    #         final_comb_embedding = comb_embedding
    #     elif self.alpha_mode == 2:
    #         final_comb_embedding = comp_comb_embedding
    #     elif self.alpha_mode == 3:
    #         final_comb_embedding = null_embedding


    #     # Compute and reshape combined embeddings
    #     X = torch.cat((cont_cate_embedding.reshape(batch_size, -1), 
    #         final_comb_embedding.reshape(batch_size, -1)), 1)\
    #         .type(torch.FloatTensor).to(self.device)

        
    #     # Calculate L2
    #     L2 = torch.pow(X, 2) * self.lamb
    #     L2 = L2.sum()

    #     return L2



##### Model and Alpha Mode #####
# Model: DNN_cart
#   0: using pre-searched architecture, 
#       feature combinations including cartesian product, IPNN or null
#   1: only using cartesian product to model feature combination
#   2: only using original embedding to compute feature combination
#   3: do not model feature combination, equal to FNN
# Model: IPNN
# Model: FNN
# Model: FM
# Model: PIN
##### ========== #####


def getmodel(model_name, cont_field, cate_field, cate_cont_feature, comb_feature,
            comb_field=0, arch=0, orig_embedding_dim=40, comb_embedding_dim=40,
            hidden_dims=[500,500,500,500,500], device=torch.device('cpu'), alpha_mode=1,
            lamb=0., selected_pairs=None, id_offsets=None):
    if model_name == 'DNN_cart':
        model = DNN_cart(cont_field, cate_field, comb_field, cate_cont_feature, 
            comb_feature, arch=arch, orig_embedding_dim=orig_embedding_dim, 
            comb_embedding_dim=comb_embedding_dim, hidden_dims=hidden_dims, 
            device=device, alpha_mode=alpha_mode, selected_pairs=selected_pairs,
            lamb=lamb)
    elif model_name == 'LR':
        model = LR(cont_field, cate_field, cate_cont_feature, device=device, lamb=lamb)
    elif model_name == 'FM':
        model = FM(cont_field, cate_field, cate_cont_feature, device=device,
            orig_embedding_dim=orig_embedding_dim, hidden_dims=hidden_dims, lamb=lamb)
    elif model_name == 'Poly2':
        model = Poly2(cont_field, cate_field, comb_field, cate_cont_feature, 
            comb_feature, device=device, lamb=lamb)
    elif model_name == 'IPNN':
        model = IPNN(cont_field, cate_field, cate_cont_feature, device=device,
            orig_embedding_dim=orig_embedding_dim, hidden_dims=hidden_dims, lamb=lamb)
    elif model_name == 'FNN':
        model = FNN(cont_field, cate_field, cate_cont_feature, device=device,
            orig_embedding_dim=orig_embedding_dim, hidden_dims=hidden_dims, lamb=lamb)
    elif model_name == 'DeepFM':
        model = DeepFM(cont_field, cate_field, cate_cont_feature, device=device,
            orig_embedding_dim=orig_embedding_dim, hidden_dims=hidden_dims, lamb=lamb)
    elif model_name == 'PIN':
        model = PIN(cont_field, cate_field, cate_cont_feature, device=device,
            orig_embedding_dim=orig_embedding_dim, hidden_dims=hidden_dims, lamb=lamb)
    
    return model
