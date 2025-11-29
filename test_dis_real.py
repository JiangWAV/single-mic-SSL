import os
import toml
import torch
from collections import OrderedDict
from tqdm import tqdm
import datetime
import torchaudio
from models.single_SSL_model import disesti_3  

# real audio 
from utils.record_audio import MyDataset   

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
    config = toml.load('./configs/real_audio/config.toml')
    checkpoint_path = os.getcwd()+ '/output/real_world/YOUR-MODEL-FILE'

    # resample
    resampler = torchaudio.transforms.Resample(orig_freq=config['test_gene_setting']['sample_rate'],
                                            new_freq=config['test_gene_setting']['target_sample_rate'])
    resampler.to(device)

    test_dataset = MyDataset(**config['FFT'], **config['test_gene_setting'])
    test_dataloader = torch.utils.data.DataLoader(test_dataset, 
                                                **config['test_dataloader'])
    
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
        input_audio = data[0].to(device)              # [b, wav length]
        target_dis = data[1].unsqueeze(-1).to(device) # [b, 1]
        
        # model
        output = model(input_audio, 0)                # [b, 1]  
        
        # data
        total_truth.append(target_dis.item())
        total_esti.append(output.item())
            
        # error
        error = torch.mean(torch.abs(output - target_dis))
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


