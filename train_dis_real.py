import os
import toml
import torch
import torch.nn as nn
from models.single_SSL_model import disesti_3     

# real audio 
from utils.record_audio import MyDataset   
import trainers.real_audio_trainer as trainer_run 

seed = 7
torch.manual_seed(seed)

def run(config, device):
    # train dataset
    train_dataset = MyDataset(**config['FFT'], **config['train_gene_setting'])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                    **config['train_dataloader'])
    
    # val dataset
    validation_dataset = MyDataset(**config['FFT'], **config['val_gene_setting'])
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, 
                                                        **config['validation_dataloader'])
    
    # loss
    loss = nn.MSELoss()


    model = disesti_3(device, **config['FFT']).to(device)
    
    # more gpus
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model).to(device)
    else:
        model.to(device)
    
    # optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config['optimizer']['lr'])

    # trainer
    trainer = trainer_run.Trainer(config=config, 
                                  model=model,
                                  optimizer=optimizer, 
                                  loss_func=loss, 
                                  train_dataset=train_dataset,
                                  train_dataloader=train_dataloader, 
                                  validation_dataset=validation_dataset,
                                  validation_dataloader=validation_dataloader, 
                                  device=device)

    # train
    trainer.train()

if __name__ == '__main__':
    os.environ['PATH'] = '/sbin:' + os.environ.get('PATH', '')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # config
    config = toml.load('./configs/real_audio/config.toml')
    run(config, device)
    