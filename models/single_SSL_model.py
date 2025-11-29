import os
import toml
import torch
import torch.nn as nn
import torch.nn.functional as F

class preprocess(nn.Module):
    def __init__(self,                 
                n_fft: int,
                hop_length: int,
                win_length: int, 
                press,
                **kwargs):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        
        self.press = press
        
    def forward(self, x, spec_type = "complex"):
        """
        :param x: [B, wave length]
        :return: [B, F, T] complex
        """
        # normalization
        x = x / (x.abs().max(dim=1, keepdim=True)[0] + 1e-8)
        
        # Padding: Use reflect padding to ensure that STFT can handle edges
        pad = (self.n_fft - self.hop_length) // 2
        x = x.unsqueeze(1)  # [B, 1, wave_length]
        x = F.pad(x, (pad, pad), mode='reflect')
        x = x.squeeze(1)                                             # [B, wave_length + 2*pad]
        
        # stft
        spec_ori = torch.stft(x, 
                            n_fft = self.n_fft, 
                            hop_length = self.hop_length, 
                            win_length = self.win_length, 
                            window = torch.hann_window(self.n_fft).to(x.device).type(x.type()), 
                            return_complex=True)
        
        # compress complex
        if spec_type == "complex":
            if self.press == "log":
                spec = torch.log(torch.clamp(spec_ori.abs(), min=1e-5)) * torch.exp(1j * spec_ori.angle())  # [B, F, T], complex
            else:
                spec = torch.pow(spec_ori.abs(), self.press) * torch.exp(1j * spec_ori.angle())  # [B, F, T], complex
        elif spec_type == "amplitude":
            if self.press == "log":
                spec = torch.log(torch.clamp(spec_ori.abs(), min=1e-5))   # [B, F, T], complex
            else:
                spec = torch.pow(spec_ori.abs(), self.press)  # [B, F, T], complex
        return spec

class LayerNormalization4D(nn.Module):
    def __init__(self, input_dimension, eps: float = 1e-5):
        super(LayerNormalization4D, self).__init__()
        assert len(input_dimension) == 2
        param_size = [1, input_dimension[0], 1, input_dimension[1]]

        self.dim = (1, 3) if param_size[-1] > 1 else (1,)
        self.gamma = nn.Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = nn.Parameter(torch.Tensor(*param_size).to(torch.float32))
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)
        self.eps = eps

    def forward(self, x: torch.Tensor):
        mu_ = x.mean(dim=self.dim, keepdim=True)
        std_ = torch.sqrt(x.var(dim=self.dim, unbiased=False, keepdim=True) + self.eps)
        x_hat = ((x - mu_) / std_) * self.gamma + self.beta
        return x_hat

class selfatten_conv(nn.Module):
    def __init__(self, in_chan, hid_chan):
        super().__init__()
        self.conv = nn.Conv2d(in_chan, hid_chan, kernel_size=1)
        self.act = nn.PReLU()
        self.norm = LayerNormalization4D((hid_chan, 1))
        
    def forward(self, x):
        output = self.conv(x)
        output = self.act(output)
        output = self.norm(output)
        return output

class selfatten(nn.Module):
    def __init__(self, ch, n_head = 4, hid_chan = 4):
        super().__init__()
        self.in_chan = ch
        self.n_head = n_head
        self.hid_chan = hid_chan

        self.Queries = nn.ModuleList()
        self.Keys = nn.ModuleList()
        self.Values = nn.ModuleList()
        
        for _ in range(self.n_head):
            self.Queries.append(
                selfatten_conv(self.in_chan, self.hid_chan)   
                )
            self.Keys.append(
                selfatten_conv(self.in_chan, self.hid_chan)   
                )
            self.Values.append(
                selfatten_conv(self.in_chan, self.in_chan // self.n_head)   
                )
            
        self.attn_concat_proj = selfatten_conv(self.in_chan, self.in_chan)
        

    def forward(self, x):
        """
        input: [B, C, F, T]
        """
        x = x.transpose(-2, -1).contiguous() # [B, C, T, F], [B, channel, F', T]
        
        batch_size, _, time, freq = x.size()
        residual = x
        
        all_Q = [q(x) for q in self.Queries]  # [B, E, T, F]
        all_K = [k(x) for k in self.Keys]  # [B, E, T, F]
        all_V = [v(x) for v in self.Values]  # [B, C/n_head, T, F]
        
        Q = torch.cat(all_Q, dim=0)  # [B', E, T, F]    B' = B*n_head
        K = torch.cat(all_K, dim=0)  # [B', E, T, F]
        V = torch.cat(all_V, dim=0)  # [B', C/n_head, T, F]
        
        Q = Q.transpose(1, 2).flatten(start_dim=2)  # [B', T, E*F]
        K = K.transpose(1, 2).flatten(start_dim=2)  # [B', T, E*F]
        V = V.transpose(1, 2)  # [B', T, C/n_head, F]
        old_shape = V.shape
        V = V.flatten(start_dim=2)  # [B', T, C*F/n_head]
        emb_dim = Q.shape[-1]  # C*F/n_head
        
        attn_mat = torch.matmul(Q, K.transpose(1, 2)) / (emb_dim**0.5)  # [B', T, T]
        attn_mat = F.softmax(attn_mat, dim=2)  # [B', T, T]
        V = torch.matmul(attn_mat, V)  # [B', T, C*F/n_head]
        V = V.reshape(old_shape)  # [B', T, C/n_head, F]
        V = V.transpose(1, 2)  # [B', C/n_head, T, F]
        emb_dim = V.shape[1]  # C/n_head

        x = V.view([self.n_head, batch_size, emb_dim, time, freq])  # [n_head, B, C/n_head, T, F]
        x = x.transpose(0, 1).contiguous()  # [B, n_head, C/n_head, T, F]

        x = x.view([batch_size, self.n_head * emb_dim, time, freq])  # [B, C, T, F]
        x = self.attn_concat_proj(x)  # [B, C, T, F]

        x = x + residual
        x = x.transpose(-2, -1).contiguous() # [B, C, F, T]
        
        out = x
        return out

class filterblock_3(nn.Module):
    def __init__(self, ch):
        super().__init__()
        # conv2d
        self.conv1 = nn.Sequential(nn.Conv2d(ch, ch, kernel_size=(1, 1)),
                                   nn.BatchNorm2d(ch),
                                   nn.PReLU())
        # filters
        self.kernel = [[1, 3], [3, 7], [7, 15]]
        self.filters = nn.ModuleList([nn.Conv2d(ch, ch, kernel_size=k, stride=(1, 1), padding=((k[0]-1)//2, (k[1]-1)//2), groups = ch) 
                                      for k in self.kernel])
        # conv2d
        self.conv2 = nn.Sequential(nn.Conv2d(len(self.kernel) * ch, ch, kernel_size=(1, 1), groups = ch),
                                   nn.BatchNorm2d(ch),
                                   nn.PReLU())
        
    def forward(self, x):
        residual = x.clone()                # 0.2s: [32, 16, 29]
        # conv2d 1
        h = self.conv1(x)
        # filters
        f_outputs = [conv(h) for conv in self.filters]
        f_outputs = torch.stack(f_outputs, dim=2)    # [B, C, 2, F, T]
        f_outputs = f_outputs.reshape(f_outputs.shape[0], -1, f_outputs.shape[3], f_outputs.shape[4]) 
        # conv2d 2
        h = self.conv2(f_outputs)
        out = h +  residual
        return out  
   
class basicblock_3(nn.Module):
    def __init__(self, in_chan, hid_chan):
        super().__init__()
        self.convs = filterblock_3(in_chan)
        self.att = selfatten(ch = in_chan, n_head = 4, hid_chan = 4)
        self.norm = LayerNormalization4D((in_chan, 1))
        
    def forward(self, x):
        x = self.convs(x)                   # [B, k subband , ch sub F, T]
        x = self.att(x)
        x = self.norm(x)
        return x
    
class disesti_3(nn.Module):
    def __init__(self,
                device,
                n_fft: int,
                hop_length: int,
                win_length: int, 
                **kwargs):
        super().__init__()
        # audio processing
        self.std = True
        self.press_coeff = 1/2
        self.eps = 1e-8
        
        # fft
        self.preprocess = preprocess(n_fft, hop_length, win_length, press = 0.5)  # amplitude, complex, 0.5
        self.n_freqs = int(n_fft // 2 + 1)    
        
        # sub band
        self.band_width = [16]*16
        self.nband = len(self.band_width)
        print(self.band_width)
        
        self.feature_dim = 32
        
        # initial conv
        self.conv1 = nn.Conv2d(self.nband * 4, self.feature_dim, kernel_size=(1,1))
        self.bn = nn.BatchNorm2d(self.feature_dim)
        self.PReLU = nn.PReLU()
        
        # basic block
        num_blocks = 4
        self.blocks = nn.ModuleList([basicblock_3(in_chan = self.feature_dim, hid_chan = 4) for _ in range(num_blocks)])

        # output
        self.conv_out = nn.Conv2d(self.feature_dim, 1, kernel_size=(1, 1), stride=(1, 1))
        self.gru = nn.GRU(input_size = 16, hidden_size = self.feature_dim, num_layers = 1, batch_first = True)
        self.linear_out = nn.Linear(self.feature_dim, 1)
        
    def forward(self, x, info=0):
        # stft
        batch_size, nsample = x.shape
        spec = self.preprocess(x, spec_type = "complex")             # [B, F, T] complex/amplitude, 0.032s: 1bin
        phase_sin = torch.sin(torch.angle(spec.real + 1j*spec.imag)) # [B, F, T]
        phase_cos = torch.cos(torch.angle(spec.real + 1j*spec.imag)) # [B, F, T]
        spec_RI   = torch.stack([spec.real, spec.imag, phase_sin, phase_cos], 1)  # B*nch, 2, F, T
        
        subband_spec_RI = []
        subband_spec = []
        band_idx = 0
        for i in range(len(self.band_width)):
            subband_spec_RI.append(spec_RI[:,:,band_idx:band_idx+self.band_width[i]].contiguous())
            subband_spec.append(spec[:,band_idx:band_idx+self.band_width[i]])  # B*nch, BW, T
            band_idx += self.band_width[i]
        
        # conv1 
        subband_specs = torch.stack(subband_spec_RI, dim = 1) # [B, num subband, 4 c, F', T] [B, 16, 4, 16, 254]
        subband_specs = subband_specs.reshape(batch_size, -1, subband_specs.size(-2), subband_specs.size(-1)) # [B, 64, 16, 254]
        x = self.conv1(subband_specs)
        x = self.bn(x)
        x = self.PReLU(x)
        
        # blocks
        for block in self.blocks:
            x = block(x)
            
        # output
        x    = self.conv_out(x)                 # [B, 1, F, T]
        x    = x.squeeze(1).transpose(-1,-2)    # [B, T, features]
        h, _ = self.gru(x)
        out  = self.linear_out(h).squeeze(-1) # [B, T]
        out_all = F.relu(out)
        out = out_all.mean(dim=-1, keepdim = True) # [B, 1]
        
        if info == 1:
            return out, out_all
        else:
            return out
        
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = toml.load('./configs/simulation/config_eval.toml')
      
    model = disesti_3(device, **config['FFT']).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total_params',total_params)
    
    # model 
    batch = 1
    x = torch.randn(batch, 3200).to(device)
    
    out = model(x, 0)
    # para: 75679

    from thop import profile
    import time

    macs, params = profile(model, inputs=(x,))
    print("MACs: ", macs)
    print("Params:", params)

    model.eval()
    input_data = torch.randn(1, 3200).to(device)

    num_runs = 100
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_data)
            # If a GPU is used, synchronization is required to ensure that each call is completed
            if device.type == 'cuda':
                torch.cuda.synchronize()
    end_time = time.time()
    average_time = (end_time - start_time) / num_runs
    print(f"Average inference time: {average_time*1000:.2f} ms")