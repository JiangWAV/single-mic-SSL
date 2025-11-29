import os
import toml
import torch
from collections import OrderedDict
from tqdm import tqdm
import datetime
import torchaudio
from models.single_SSL_model import disesti_3

# simulation
from utils.simulation import MyDataset

def _data_ready(data):
    # data = (target_reverb, dis_info)
    input_audio = data[0].to(device).squeeze(1) # [b, wav length]
    if resample:
        # [batch, channels, wav_length]
        input_audio = resampler(input_audio.unsqueeze(1))  # [b, 1, new_wav_length]
        input_audio = input_audio.squeeze(1)               # [b, new_wav_length]           
    
    # distance
    dis_info = data[1]
    srcs_dis_value = dis_info['srcs_dis_value'].to(device) # [b, n_src]     
    datas = {'input': input_audio,
                'target_distance': srcs_dis_value,         # [b, 1]
                }
    return datas

def remove_module_prefix(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_k = k[len("module."):]
        else:
            new_k = k
        new_state_dict[new_k] = v
    return new_state_dict

if __name__ == '__main__':
    os.environ['PATH'] = '/sbin:' + os.environ.get('PATH', '')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # configs
    config = toml.load('./configs/simulation/config.toml')
    #　test
    # 1 room
    checkpoint_path = os.getcwd()+ '/output/1room/YOUR-MODEL-FILE*.tar'

    # 100 room
    # checkpoint_path = os.getcwd()+ '/output/100room/YOUR-MODEL-FILE'

    # resample
    resampler = torchaudio.transforms.Resample(orig_freq=config['test_gene_setting']['sample_rate'],
                                            new_freq=config['test_gene_setting']['target_sample_rate'])
    resampler.to(device)

    test_dataset = MyDataset(**config['test_dataset'],**config['FFT'], **config['test_gene_setting'])
    test_dataloader = torch.utils.data.DataLoader(test_dataset, 
                                                **config['test_dataloader'],
                                                collate_fn=test_dataset.collate_fn)



    model = disesti_3(device, **config['FFT']).to(device)
    model.to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_weights = checkpoint['model']
    model_weights = remove_module_prefix(model_weights)
    model.load_state_dict(model_weights, strict=True)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total_params',total_params)

    data_sample_rate = config['test_gene_setting']['sample_rate']
    target_sample_rate = config['test_gene_setting']['target_sample_rate']
    resample = False
    if data_sample_rate != target_sample_rate:
        resample = True
        resampler = torchaudio.transforms.Resample(orig_freq=data_sample_rate,
                                                        new_freq=target_sample_rate)
        resampler.to(device)

    test_bar = tqdm(test_dataloader, ncols=100)
    total_error = 0
    total_num = 0
    total_truth = []
    total_esti = []

    for step, data in enumerate(test_bar):
        data_ready  = _data_ready(data)
        input_audio = data_ready['input']# [b, wav length]
        target_dis  = data_ready['target_distance'] # [b, 1]
        
        # model
        output = model(input_audio, 0) # [b, 1]  
        
        # data
        total_truth.append(target_dis.item())
        total_esti.append(output.item())
            
        # error
        error = torch.abs(output - target_dis)
        total_error += error.item()
        total_num += 1
        
        print('mean error', round(total_error/total_num, 3))
            
    mean_error = total_error/total_num

    # save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.getcwd()+ f'/test_results/results_{timestamp}.txt'
    with open(filename, 'w') as f:
        f.write(f'mean_error: {mean_error}\n')
        for truth, esti in zip(total_truth, total_esti):
            f.write(f"truth: {truth}, esti: {esti}\n")
    print(f'Results saved to {filename}')


