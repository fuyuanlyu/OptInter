import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DNN_cart_search(nn.Module):
    def __init__(self, cont_field, cate_field, comb_field, cate_cont_feature, 
                comb_feature, criterion, selected_pairs, orig_embedding_dim=32, 
                comb_embedding_dim=8, hidden_dims=[100,100], tau=5e-5,
                device=torch.device('cpu'), num_ops=3):
        super(DNN_cart_search, self).__init__()
        self.cont_field = cont_field
        self.cate_field = cate_field
        self.comb_field = comb_field
        self.cate_cont_feature = cate_cont_feature
        self.comb_feature = comb_feature
        self._criterion = criterion
        self.selected_pairs = selected_pairs
        self.orig_embedding_dim = orig_embedding_dim
        self.comb_embedding_dim = comb_embedding_dim
        self.hidden_dims = hidden_dims
        self.device = device
        self.num_ops = num_ops

        # Gumbel softmax
        self.tau = tau
        self.step = 0

        # initialize alpha
        self.initialize_alphas()
        
        # Create embedding tables
        self.orig_embeddings_table = \
            nn.Embedding(self.cate_cont_feature, self.orig_embedding_dim)
        self.comb_embeddings_table = \
            nn.Embedding(self.comb_feature, self.comb_embedding_dim)
        self.addition_embedding_table = \
            nn.Embedding(self.cate_cont_feature, self.comb_embedding_dim)
        
        # Create layers
        self.fc_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        first_layer_neurons =  self.orig_embedding_dim * \
            (self.cate_field + self.cont_field) + self.comb_field * self.comb_embedding_dim 
        
        self.fc_layers.append(nn.Linear(first_layer_neurons, hidden_dims[0]))
        for _, (in_size, out_size) in enumerate(zip(hidden_dims[:-1], hidden_dims[1:])):
            self.fc_layers.append(nn.Linear(in_size, out_size))

        for _, size in enumerate(hidden_dims):
            self.norm_layers.append(nn.LayerNorm(size))

        self.output_layer = nn.Linear(hidden_dims[-1], 1)

    def forward(self, conts, cates, combs):
        # Assert the batch sizes are the same
        batch_size = conts.size()[0]
        assert batch_size == cates.size()[0]
        assert batch_size == combs.size()[0]

        # Get continuous, categorical and free combinad embeddings
        cont_embedding = torch.IntTensor(np.arange(self.cont_field))\
                .expand_as(conts).to(self.device)
        cont_embedding = self.orig_embeddings_table(cont_embedding)
        conts = conts.unsqueeze(2)
        cont_embedding = torch.mul(cont_embedding, conts)
        cate_embedding = self.orig_embeddings_table(cates)
        comb_embedding = self.comb_embeddings_table(combs)

        # Compute combined embeddings
        addition_cate_embedding = self.addition_embedding_table(cates)
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
        null_embedding = torch.zeros_like(comb_embedding).to(self.device)
       
        # Get normalized alpha
        t = torch.tensor(max(0.01, 1-self.tau*self.step), dtype=torch.float32, device=self.device)
        normalized_alpha = F.softmax(self.arch_parameters / t, dim=1)
        self.step += 1

        # Compute final combined embedding
        final_comb_embedding = comb_embedding.mul(normalized_alpha[:,0].unsqueeze(0).unsqueeze(2)) \
                        + comp_comb_embedding.mul(normalized_alpha[:,1].unsqueeze(0).unsqueeze(2)) \
                        + null_embedding.mul(normalized_alpha[:,2].unsqueeze(0).unsqueeze(2))

        # Reshape all embeddings
        cont_embedding = cont_embedding.reshape(batch_size, -1)
        cate_embedding = cate_embedding.reshape(batch_size, -1)
        final_comb_embedding = final_comb_embedding.reshape(batch_size, -1)
        
        # Compute final X as model input
        X = torch.cat((cont_embedding, cate_embedding, final_comb_embedding), 1)\
            .type(torch.FloatTensor).to(self.device)

        # Pass to FC layers
        for idx in range(len(self.fc_layers)):
            X = self.fc_layers[idx](X)
            X = self.norm_layers[idx](X)
            X = F.relu(X)
        logit = self.output_layer(X)
        logit = torch.sigmoid(logit)

        return logit

    def new(self):
        model_new = DNN(self.cont_field, self.cate_field, self.comb_field,
                    self.cate_cont_feature, self.comb_feature, 
                    self._criterion, self.selected_pairs, device=self.device,
                    orig_embedding_dim=self.orig_embedding_dim,
                    comb_embedding_dim=self.comb_embedding_dim,
                    hidden_dims=self.hidden_dims,
                    num_ops = self.num_ops).to(self.device)
        for x, y in zip(model_new.arch_parameters, self.arch_parameters):
            x.data.copy_(y.data)
        return model_new

    def _loss(self, features, target):
        logits = self.forward(features[0], features[1], features[2])
        return self._criterion(logits, target) 

    def initialize_alphas(self):
        self.arch_parameters = torch.empty(self.comb_field, self.num_ops, \
                        device=self.device, requires_grad=True)
        nn.init.xavier_normal_(self.arch_parameters)

    def get_arch_parameters(self):
        return self.arch_parameters.cpu().detach()


class DNN_cart_bi_lvl(nn.Module):
    def __init__(self, cont_field, cate_field, comb_field, cate_cont_feature, 
                comb_feature, criterion, selected_pairs, orig_embedding_dim=32, 
                comb_embedding_dim=8, hidden_dims=[100,100], 
                device=torch.device('cpu')):
        super(DNN_cart_bi_lvl, self).__init__()
        self.cont_field = cont_field
        self.cate_field = cate_field
        self.comb_field = comb_field
        self.cate_cont_feature = cate_cont_feature
        self.comb_feature = comb_feature
        self._criterion = criterion
        self.selected_pairs = selected_pairs
        self.orig_embedding_dim = orig_embedding_dim
        self.comb_embedding_dim = comb_embedding_dim
        self.hidden_dims = hidden_dims
        self.device = device
        
        # Create embedding tables
        self.orig_embeddings_table = \
            nn.Embedding(self.cate_cont_feature, self.orig_embedding_dim)
        self.comb_embeddings_table = \
            nn.Embedding(self.comb_feature, self.comb_embedding_dim)
        self.addition_embedding_table = \
            nn.Embedding(self.cate_cont_feature, self.comb_embedding_dim)
        
        # Create layers
        self.fc_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        first_layer_neurons =  self.orig_embedding_dim * \
            (self.cate_field + self.cont_field) + self.comb_field * self.comb_embedding_dim 
        
        self.fc_layers.append(nn.Linear(first_layer_neurons, hidden_dims[0]))
        for _, (in_size, out_size) in enumerate(zip(hidden_dims[:-1], hidden_dims[1:])):
            self.fc_layers.append(nn.Linear(in_size, out_size))

        for _, size in enumerate(hidden_dims):
            self.norm_layers.append(nn.LayerNorm(size))

        self.output_layer = nn.Linear(hidden_dims[-1], 1)

    def forward(self, conts, cates, combs, alpha):
        # Assert the batch sizes are the same
        batch_size = conts.size()[0]
        assert batch_size == cates.size()[0]
        assert batch_size == combs.size()[0]

        # Get continuous, categorical and free combinad embeddings
        cont_embedding = torch.IntTensor(np.arange(self.cont_field))\
                .expand_as(conts).to(self.device)
        cont_embedding = self.orig_embeddings_table(cont_embedding)
        conts = conts.unsqueeze(2)
        cont_embedding = torch.mul(cont_embedding, conts)
        cate_embedding = self.orig_embeddings_table(cates)
        comb_embedding = self.comb_embeddings_table(combs)

        # Compute combined embeddings
        addition_cate_embedding = self.addition_embedding_table(cates)
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
        null_embedding = torch.zeros_like(comb_embedding).to(self.device)
       
        # Get normalized alpha
        normalized_alpha = F.softmax(alpha, dim=1)

        # Compute final combined embedding
        final_comb_embedding = comb_embedding.mul(normalized_alpha[:,0].unsqueeze(0).unsqueeze(2)) \
                        + comp_comb_embedding.mul(normalized_alpha[:,1].unsqueeze(0).unsqueeze(2)) \
                        + null_embedding.mul(normalized_alpha[:,2].unsqueeze(0).unsqueeze(2))

        # Reshape all embeddings
        cont_embedding = cont_embedding.reshape(batch_size, -1)
        cate_embedding = cate_embedding.reshape(batch_size, -1)
        final_comb_embedding = final_comb_embedding.reshape(batch_size, -1)
        
        # Compute final X as model input
        X = torch.cat((cont_embedding, cate_embedding, final_comb_embedding), 1)\
            .type(torch.FloatTensor).to(self.device)

        # Pass to FC layers
        for idx in range(len(self.fc_layers)):
            X = self.fc_layers[idx](X)
            X = self.norm_layers[idx](X)
            X = F.relu(X)
        logit = self.output_layer(X)
        logit = torch.sigmoid(logit)

        return logit

    def new(self):
        model_new = DNN_cart_bi_lvl(self.cont_field, self.cate_field, self.comb_field,
                    self.cate_cont_feature, self.comb_feature, self._criterion,
                    selected_pairs=self.selected_pairs, 
                    hidden_dims=self.hidden_dims, device=self.device,
                    orig_embedding_dim=self.orig_embedding_dim,
                    comb_embedding_dim=self.comb_embedding_dim).to(self.device)
        for x, y in zip(model_new.parameters(), self.parameters()):
            x.data.copy_(y.data)
        return model_new

    def _loss(self, features, target, alpha):
        logits = self.forward(features[0], features[1], features[2], alpha)
        return self._criterion(logits, target) 

