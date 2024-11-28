import numpy as np
import torch
import torch.nn as nn
from torch import optim

from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers import BertTokenizer, BertModel
from einops import rearrange
from embed import DataEmbedding, DataEmbedding_wo_time
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

class GateNetwork(nn.Module):
    def __init__(self, input_dim, enc_in):
        super(GateNetwork, self).__init__()
        self.gate_layer = nn.Linear(input_dim, enc_in)

    def forward(self, x):
        # Gate output as weights for experts
        return torch.softmax(self.gate_layer(x), dim=-1)

class GPT4TS(nn.Module):
    
    def __init__(self, configs, device):
        super(GPT4TS, self).__init__()
        self.is_gpt = configs.is_gpt
        self.patch_size = configs.patch_size
        self.pretrain = configs.pretrain
        self.stride = configs.stride
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_num += 1

        self.enc_in = configs.enc_in
        self.gate = GateNetwork(configs.seq_len,self.enc_in)
        
        if configs.is_gpt:
            if configs.pretrain:
                self.gpt2 = GPT2Model.from_pretrained('/root/Load_LLM-experiments/Mine_4/GPT4TS/models/local_gpt2/', output_attentions=True, output_hidden_states=True)  # loads a pretrained GPT-2 base model
            else:
                print("------------------no pretrain------------------")
                self.gpt2 = GPT2Model(GPT2Config())
            self.gpt2.h = self.gpt2.h[:configs.gpt_layers]
            print("gpt2 = {}".format(self.gpt2))
        
        self.in_layer = nn.Linear(configs.patch_size, configs.d_model)
        self.out_layer = nn.Linear(configs.d_model * self.patch_num, configs.pred_len)
        
        if configs.freeze and configs.pretrain:
            for i, (name, param) in enumerate(self.gpt2.named_parameters()):
                if 'ln' in name or 'wpe' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        for layer in (self.gpt2, self.in_layer, self.out_layer,self.gate):
            layer.to(device=device)
            layer.train()
        
        self.cnt = 0


    def forward(self, x, x_mark, itr):
        B, L, M = x.shape

        expert_outputs = []
        for i in range(self.enc_in):
            x_c = x[:, :, i, None]
            
            means = x_c.mean(1, keepdim=True).detach()
            x_c = x_c - means
            stdev = torch.sqrt(torch.var(x_c, dim=1, keepdim=True, unbiased=False)+ 1e-5).detach() 
            x_c /= stdev

            x_c = rearrange(x_c, 'b l m -> b m l')

            x_c = self.padding_patch_layer(x_c)
            x_c = x_c.unfold(dimension=-1, size=self.patch_size, step=self.stride)
            x_c = rearrange(x_c, 'b m n p -> (b m) n p')

            outputs_c = self.in_layer(x_c)
            if self.is_gpt:
                outputs_c = self.gpt2(inputs_embeds=outputs_c).last_hidden_state

            outputs_c=outputs_c.reshape(B*1, -1)
            outputs_c = self.out_layer(outputs_c)
            # outputs_1 = self.out_layer(outputs_1.reshape(B*M, -1))
            outputs_c = rearrange(outputs_c, '(b m) l -> b l m', b=B)

            outputs_c = outputs_c * stdev
            outputs_c = outputs_c + means

            expert_outputs.append(outputs_c)

        expert_outputs = torch.stack(expert_outputs, dim=1)  # Shape: (B, enc_in, pred_len, M)

        # Gating logic
        # Flatten the input for gating network: (B, L, M) -> (B, L*M)
        gate_input = rearrange(x[:,:,i,None], 'b l m -> b (l m)')
        gate_weights = self.gate(gate_input)  # Shape: (B, enc_in)
        # print("gate_weights_shape",gate_weights.shape)

        # Expand gate weights for broadcasting: (B, num_experts) -> (B, enc_in, 1, 1)
        gate_weights = gate_weights.unsqueeze(-1).unsqueeze(-1)

        # Weighted sum of expert outputs using gate weights
        output = (gate_weights * expert_outputs).sum(dim=1)

        return output
