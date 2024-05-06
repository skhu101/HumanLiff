import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import cv2

# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

def to_cuda(device, sp_input, tp_input=None):
    for key in sp_input.keys():
        if torch.is_tensor(sp_input[key]):
            sp_input[key] = sp_input[key].to(device)
        if key=='params':
            for key1 in sp_input['params']:
                if torch.is_tensor(sp_input['params'][key1]):
                    sp_input['params'][key1] = sp_input['params'][key1].to(device)
        if key=='t_params':
            for key1 in sp_input['t_params']:
                if torch.is_tensor(sp_input['t_params'][key1]):
                    sp_input['t_params'][key1] = sp_input['t_params'][key1].to(device)

    if tp_input==None:
        return sp_input
    
    for key in tp_input.keys():
        if torch.is_tensor(tp_input[key]):
            tp_input[key] = tp_input[key].to(device) 
        if key=='params':
            for key1 in tp_input['params']:
                if torch.is_tensor(tp_input['params'][key1]):
                    tp_input['params'][key1] = tp_input['params'][key1].to(device)
        if key=='t_params':
            for key1 in tp_input['t_params']:
                if torch.is_tensor(tp_input['t_params'][key1]):
                    tp_input['t_params'][key1] = tp_input['t_params'][key1].to(device)

    return sp_input, tp_input


class PositionalEncoding(torch.nn.Module):
    """
    Implement NeRF's positional encoding
    """

    def __init__(self, num_freqs=6, d_in=3, freq_factor=np.pi, include_input=True):
        super().__init__()
        self.num_freqs = num_freqs
        self.d_in = d_in
        self.freqs = 2.**torch.linspace(0., num_freqs-1, steps=num_freqs)
        self.d_out = self.num_freqs * 2 * d_in
        self.include_input = include_input
        if include_input:
            self.d_out += d_in
        # f1 f1 f2 f2 ... to multiply x by
        self.register_buffer(
            "_freqs", torch.repeat_interleave(self.freqs, 2).view(1, -1, 1)
        )
        # 0 pi/2 0 pi/2 ... so that
        # (sin(x + _phases[0]), sin(x + _phases[1]) ...) = (sin(x), cos(x)...)
        _phases = torch.zeros(2 * self.num_freqs)
        _phases[1::2] = np.pi * 0.5
        self.register_buffer("_phases", _phases.view(1, -1, 1))

    def forward(self, x):
        """
        Apply positional encoding (new implementation)
        :param x (batch, self.d_in)
        :return (batch, self.d_out)
        """
        # with profiler.record_function("positional_enc"):
        embed = x.unsqueeze(1).repeat(1, self.num_freqs * 2, 1)
        embed = torch.sin(torch.addcmul(self._phases, embed, self._freqs))
        if x.shape[0]==0:
            embed = embed.view(x.shape[0], self.num_freqs*6)
        else:
            embed = embed.view(x.shape[0], -1)
            
        if self.include_input:
            embed = torch.cat((x, embed), dim=-1)
        return embed