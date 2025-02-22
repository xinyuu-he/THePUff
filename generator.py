import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Generator(nn.Module):
    def __init__(self, node_type, N, node_embs, device, args):
        super(Generator, self).__init__()
        self.node_type = node_type
        self.N = N
        self.batch_size = args.batch_size
        self.hidden_units = args.hidden_units
        self.noise_dim = args.noise_dim
        self.max_path_len = args.max_path_len
        self.W_down_generator_size = args.W_down_generator_size
        self.num_G_layer = args.num_G_layer
        # include node type to be deleted
        self.node_classes = args.node_classes + 1
        self.node_embs = node_embs
        self.device = device
        self.node_emb_size = 128

        self.type_0 = torch.tensor(node_type[0], dtype=torch.float).to(self.device)
        self.type_1 = torch.tensor(node_type[1], dtype=torch.float).to(self.device)
        self.type_2 = torch.tensor(node_type[2], dtype=torch.float).to(self.device)

        self.lin_node_0 = nn.Linear(self.hidden_units, self.node_emb_size)
        self.lin_node_1 = nn.Linear(self.hidden_units, self.node_emb_size)
        self.lin_node_2 = nn.Linear(self.hidden_units, self.node_emb_size)

        if self.node_classes == 5:
            self.type_3 = torch.tensor(node_type[3], dtype=torch.float).to(self.device)
            self.lin_node_3 = nn.Linear(self.hidden_units, self.node_emb_size)

        self.W_down_generator_type = nn.Linear(self.node_classes, self.W_down_generator_size)
        self.W_down_generator_node = nn.Linear(self.N + 1, self.W_down_generator_size)

        self.lstm = nn.LSTM(self.hidden_units, self.hidden_units, self.num_G_layer)
        self.init_lin_1 = nn.Linear(self.noise_dim, self.hidden_units)
        self.init_lin_2_h = nn.Linear(self.hidden_units, self.hidden_units)
        self.init_lin_2_c = nn.Linear(self.hidden_units, self.hidden_units)
        self.lin_node_type = nn.Linear(self.hidden_units, self.node_classes)

    def reverse_sampling(self, dist):
        dist_weight = torch.exp(-dist)
        return torch.multinomial(dist_weight, 1)[0]

    def forward(self, z):
        outputs_type, outputs_node, init_c, init_h = [], [], [], []
        outputs_idx = []
        for _ in range(self.num_G_layer):
            intermediate = torch.tanh(self.init_lin_1(z))
            init_c.append(torch.tanh(self.init_lin_2_c(intermediate)))
            init_h.append(torch.tanh(self.init_lin_2_h(intermediate)))
        # Initialize an input tensor
        inputs = Variable(torch.zeros((self.batch_size, self.hidden_units))).to(self.device)
        hidden = (torch.stack(init_c, dim=0), torch.stack(init_h, dim=0))

        # LSTM time steps
        for i in range(self.max_path_len):
            out, hidden = self.lstm(inputs.unsqueeze(0), hidden)
            output_bef = self.lin_node_type(out.squeeze(0))
            output_type = F.gumbel_softmax(output_bef, dim=1, tau=3, hard=True)
            temp_node = []
            for j, x in enumerate(torch.argmax(output_type, dim=1)):
                if x == 0:
                    temp_output_node = self.lin_node_0(out.squeeze(0)[j])
                    dist = torch.norm(self.node_embs[x.item()] - temp_output_node, dim=1, p=None)
                    dist = dist.topk(int(dist.shape[0] // 1), largest=False)
                    candidate = dist.indices[self.reverse_sampling(dist.values)]
                    temp_node.append(self.type_0[candidate])
                elif x == 1:
                    temp_output_node = self.lin_node_1(out.squeeze(0)[j])
                    dist = torch.norm(self.node_embs[x.item()] - temp_output_node, dim=1, p=None)
                    dist = dist.topk(int(dist.shape[0] // 1), largest=False)
                    candidate = dist.indices[self.reverse_sampling(dist.values)]
                    temp_node.append(self.type_1[candidate])
                elif x == 2:
                    temp_output_node = self.lin_node_2(out.squeeze(0)[j])
                    dist = torch.norm(self.node_embs[x.item()] - temp_output_node, dim=1, p=None)
                    dist = dist.topk(int(dist.shape[0] // 1), largest=False)
                    candidate = dist.indices[self.reverse_sampling(dist.values)]
                    temp_node.append(self.type_2[candidate])
                elif x == 3 and self.node_classes > 4:
                    temp_output_node = self.lin_node_3(out.squeeze(0)[j])
                    dist = torch.norm(self.node_embs[x.item()] - temp_output_node, dim=1, p=None)
                    dist = dist.topk(int(dist.shape[0] // 1), largest=False)
                    candidate = dist.indices[self.reverse_sampling(dist.values)]
                    temp_node.append(self.type_3[candidate])
                else: # x == 4:
                    temp_node.append(torch.tensor(self.N).to(self.device))
            outputs_idx.append(list(map(int, temp_node)))
            temp_node = torch.stack(temp_node)
            output_node = F.one_hot(temp_node.to(int), self.N + 1).float()
            outputs_type.append(output_type)
            outputs_node.append(output_node)

            inputs = self.W_down_generator_type(output_type) + self.W_down_generator_node(output_node)

        outputs_type = torch.stack(outputs_type, dim=1)
        outputs_node = torch.stack(outputs_node, dim=1)

        return outputs_type, outputs_node, outputs_idx