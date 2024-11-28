import numpy as np
import torch
import torch.nn as nn
from torch import optim

from transformers.models.gpt2.modeling_gpt2 import GPT2Model
# from models.hugging_gpt2.GPT2 import GPT2Model
from transformers import BertTokenizer, BertModel
from einops import rearrange
from embed import DataEmbedding, DataEmbedding_wo_time,TemporalEmbedding
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from layers.FFT_Block import FFT_Block

class Adapter(nn.Module):
    def __init__(self, in_feat, hid_dim, skip=True):
        super().__init__()
        self.D_fc1 = nn.Linear(in_feat, hid_dim)
        self.D_fc2 = nn.Linear(hid_dim, in_feat)
        self.act = nn.GELU()
        self.skip = skip
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        if self.skip:
            return x + self.drop(self.D_fc2(self.act(self.D_fc1(x))))
        else:
            return self.drop(self.D_fc2(self.act(self.D_fc1(x))))

class GateNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(GateNetwork, self).__init__()
        self.gate_layer = nn.Linear(input_dim, num_experts)

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
        self.freq = 'h' if configs.freq==1 else 'm'

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_num += 1

        # MoE
        self.num_experts = configs.num_experts
        self.sampling_rates = configs.sampling_rates[:self.num_experts]
        self.gate = GateNetwork(configs.seq_len*configs.enc_in,self.num_experts)
        
        if configs.is_gpt:
            if configs.pretrain:
                self.gpt2 = GPT2Model.from_pretrained('/root/Load_LLM-experiments/Mine_3/GPT4TS/models/local_gpt2/', output_attentions=True, output_hidden_states=True)  # loads a pretrained GPT-2 base model
            else:
                print("------------------no pretrain------------------")
                self.gpt2 = GPT2Model(GPT2Config())
            self.gpt2.h = self.gpt2.h[:configs.gpt_layers]
            # for i in range(configs.gpt_layers):
            #     # self.gpt2.h[i].scale = configs.scale
            #     self.gpt2.h[i].scale = configs.scale
            #     self.gpt2.h[i].attn.scale = configs.scale
            #     if configs.T_type == 1:
            #         self.gpt2.h[i].T_adapter = Adapter(configs.d_model, configs.adapter_dim, skip=False)
            #         self.gpt2.h[i].T_adapter_gate = torch.nn.Parameter(torch.zeros(1, self.patch_num, 1))
            #     if configs.C_type == 1:
            #         self.gpt2.h[i].C_adapter = Adapter(configs.d_model, configs.adapter_dim, skip=False)
            #         self.gpt2.h[i].C_num = configs.enc_in
            #         self.gpt2.h[i].C_adapter_gate = torch.nn.Parameter(torch.zeros(1, configs.enc_in, 1))
            print("gpt2 = {}".format(self.gpt2))

        
        self.in_layer = nn.Linear(configs.patch_size, configs.d_model)
        self.expert_output_layers = [nn.Linear(configs.d_model * (((configs.seq_len//sampling_rate) - self.patch_size) // self.stride + 1+1), configs.pred_len) for sampling_rate in self.sampling_rates]
        
        if configs.freeze and configs.pretrain:
            for i, (name, param) in enumerate(self.gpt2.named_parameters()):
                if 'ln' in name or 'wpe' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        for layer in (self.gpt2, self.in_layer,self.gate):
            layer.to(device=device)
            layer.train() 
            
        for out_layer in self.expert_output_layers:
            out_layer.to(device=device)
            out_layer.train()        
        self.cnt = 0


    def forward(self, x, x_mark,itr):
        B, L, M = x.shape

        
        expert_outputs = []
        for i,sampling_rate in enumerate(self.sampling_rates):
            x_e = x[:, ::sampling_rate, :]
            means = x_e.mean(1, keepdim=True).detach()
            x_e = x_e - means
            stdev = torch.sqrt(torch.var(x_e, dim=1, keepdim=True, unbiased=False)+ 1e-5).detach() 
            x_e /= stdev

            x_e = rearrange(x_e, 'b l m -> b m l')

            x_e = self.padding_patch_layer(x_e)
            x_e = x_e.unfold(dimension=-1, size=self.patch_size, step=self.stride)
            x_e = rearrange(x_e, 'b m n p -> (b m) n p')

            outputs_e = self.in_layer(x_e)
            if self.is_gpt:
                outputs_e = self.gpt2(inputs_embeds=outputs_e).last_hidden_state

            outputs_e=outputs_e.reshape(B*M, -1)
            out_layer = self.expert_output_layers[i]
            outputs_e = out_layer(outputs_e)
            # outputs_1 = self.out_layer(outputs_1.reshape(B*M, -1))
            outputs_e = rearrange(outputs_e, '(b m) l -> b l m', b=B)

            outputs_e = outputs_e * stdev
            outputs_e = outputs_e + means

            expert_outputs.append(outputs_e)

        expert_outputs = torch.stack(expert_outputs, dim=1)  # Shape: (B, num_experts, pred_len, M)

        # Gating logic
        # Flatten the input for gating network: (B, L, M) -> (B, L*M)
        gate_input = rearrange(x, 'b l m -> b (l m)')
        gate_weights = self.gate(gate_input)  # Shape: (B, num_experts)
        # print(gate_weights)

        # Expand gate weights for broadcasting: (B, num_experts) -> (B, num_experts, 1, 1)
        gate_weights = gate_weights.unsqueeze(-1).unsqueeze(-1)

        # Weighted sum of expert outputs using gate weights
        output = (gate_weights * expert_outputs).sum(dim=1)

        return output
