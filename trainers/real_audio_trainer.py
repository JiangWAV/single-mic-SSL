import os
from telnetlib import EL
import torch
import torchaudio
import toml
import json
from datetime import datetime
from tqdm import tqdm
from glob import glob
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(self, 
                 config, 
                 model, 
                 optimizer, 
                 loss_func,
                 train_dataset,
                 train_dataloader, 
                 validation_dataset,
                 validation_dataloader, 
                 device):
        # load model
        self.device  = device
        self.config  = config
        self.model   = model
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('total_params',total_params)
        
        # optimize
        self.optimizer = optimizer
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min', factor=0.8, patience=10,verbose=True)
        self.loss_func = loss_func.to(self.device)
        
        # data
        self.train_dataset         = train_dataset
        self.train_dataloader      = train_dataloader
        self.validation_dataset    = validation_dataset
        self.validation_dataloader = validation_dataloader
        
        # resample
        self.data_sample_rate   = config['train_gene_setting']['sample_rate']
        self.target_sample_rate = config['train_gene_setting']['target_sample_rate']
        self.resample = False
        if self.data_sample_rate != self.target_sample_rate:
            self.resample = True
            self.resampler = torchaudio.transforms.Resample(orig_freq=self.data_sample_rate,
                                                            new_freq=self.target_sample_rate)
            self.resampler.to(self.device)
            
        # dis config 
        self.n_src = config['train_gene_setting']['n_src']
        
        # training config
        self.trainer_config = config['trainer']
        self.epochs = self.trainer_config['epochs']
        self.save_checkpoint_interval = self.trainer_config['save_checkpoint_interval']
        self.clip_grad_norm_value = self.trainer_config['clip_grad_norm_value']
        self.resume = self.trainer_config['resume']
        self.resume_step = self.trainer_config['resume_step']
        
        # path                                               
        if not self.resume:
            self.exp_path = self.trainer_config['exp_path'] + datetime.now().strftime("%Y-%m-%d-%Hh%Mm") # '_'
        else:
            self.exp_path = self.trainer_config['exp_path'] + self.trainer_config['resume_datetime']
        self.log_path = os.path.join(self.exp_path, 'logs')
        self.checkpoint_path = os.path.join(self.exp_path, 'checkpoints')
        self.sample_path = os.path.join(self.exp_path, 'val_samples')
        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(self.checkpoint_path, exist_ok=True)
        os.makedirs(self.sample_path, exist_ok=True)
        
        # save the config
        with open(
            os.path.join(
                self.exp_path, 'config.toml'.format(datetime.now().strftime("%Y-%m-%d-%Hh%Mm"))), 'w') as f:
            toml.dump(config, f)

        self.writer = SummaryWriter(self.log_path)

        # setting
        self.start_epoch = 1
        self.total_step  = 1
                
    def train(self):
        timestamp_txt = os.path.join(self.exp_path, 'timestamp.txt')
        mode = 'a' if os.path.exists(timestamp_txt) else 'w'
        with open(timestamp_txt, mode) as f:
            f.write('[{}] start for {} epochs\n'.format(
                datetime.now().strftime("%Y-%m-%d-%H:%M"), self.epochs))
        initial_best_score_set = False
        if self.resume:
            self._resume_checkpoint()
        
        for epoch in range(self.start_epoch, self.epochs + self.start_epoch):
            # train
            self._set_train_mode()
            self._train_epoch(epoch)
            
            # eval
            self._set_eval_mode()
            eval_metric = self._validation_epoch(epoch)           
            # scheduler
            self.scheduler.step(eval_metric)
            if not initial_best_score_set:
                self.best_score = eval_metric
                initial_best_score_set = True
                
            # write eval metric
            with open(timestamp_txt, 'a') as f:
                f.write('[{}] Epoch {}: {}\n'.format(
                    datetime.now().strftime("%Y-%m-%d-%H:%M"), epoch, eval_metric))
                
            # Check if it is the best model and save the checkpoint
            if eval_metric <= self.best_score:
                self.best_score = eval_metric      
                # save best model
                is_best = True
                self._save_checkpoint(epoch, eval_metric, is_best=is_best)
                is_best = False
                
            # save model with interval
            if epoch % self.save_checkpoint_interval == 0:
                self._save_checkpoint(epoch, eval_metric, is_best=False)

    def _train_epoch(self, epoch):
        total_loss = 0 
        step = 0
        # set dataset's epoch
        bar = tqdm(self.train_dataloader, ncols=100)
        for step, data in enumerate(bar, 1):
            input_audio = data[0].to(self.device)              # [b, wav length]
            target_dis = data[1].unsqueeze(-1).to(self.device) # [b, 1]

            # model
            output, all_value = self.model(input_audio, info = 1) # [b, 1] , [b, time step]
            
            # loss
            loss = self.loss_func(output, target_dis)
            loss = loss.mean()
            
            # add loss
            total_loss += loss.item()  
            
            # opti
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm_value)
            self.optimizer.step()    
            # step
            step += 1

            # record
            bar.desc = '   train[{}/{}][{}]'.format(
                epoch, self.epochs + self.start_epoch-1, datetime.now().strftime("%Y-%m-%d-%H:%M"))
            bar.postfix = 'loss={:.3f}'.format(total_loss / (step+10e-6))

            # epoch record 
            self.writer.add_scalars('lr', {'lr': self.optimizer.param_groups[0]['lr']}, epoch)
            self.writer.add_scalars('train_loss', {'train_loss': total_loss/ (step+10e-6)}, epoch)
                        
    def _validation_epoch(self, epoch):
        total_loss = 0
        step = 0
        
        bar = tqdm(self.validation_dataloader, ncols=100)
        for step, data in enumerate(bar, 1):
            input_audio = data[0].to(self.device) # [b, wav length]
            target_dis = data[1].unsqueeze(-1).to(self.device) # [b, 1]

            # model 
            output = self.model(input_audio, 0)  
            
            # loss
            loss = self.loss_func(output, target_dis)
            loss = loss.mean()
            
            # add loss
            total_loss += loss.item()  
        
            # save dis info
            if step <= 3:
                info_used = []
                info_used.append({'esti dis': output.cpu().detach().numpy().tolist()})
                info_used.append({'target dis': target_dis.cpu().numpy().tolist()})

                file_path = os.path.join(self.sample_path, '{}_epoch{}_info_dis.json'.format(step, epoch))
                with open(file_path, 'w') as json_file:
                    json.dump(info_used, json_file, indent=4)
                    
            eval_metric = total_loss/(step+10e-6)
        print('validation epoch {}: loss={:.3f}'.format(epoch, eval_metric))
        return eval_metric

    def _data_ready(self, data):
        # data = (input_mix, target_reverb, target_clean, dis_info, RIR_info, mix_info, speaker_labels, longspeech)
        # input: target_reverb
        input_audio = data[1].to(self.device).squeeze(1) # [b, wav length]
        if self.resample:
            # [batch, channels, wav_length]
            input_audio = self.resampler(input_audio.unsqueeze(1))  # [b, 1, new_wav_length]
            input_audio = input_audio.squeeze(1)                    # [b, new_wav_length]           
        
        # distance
        dis_info = data[3]
        srcs_dis_value = dis_info['srcs_dis_value'].to(self.device) # [b, n_src]
        
        datas = {'input': input_audio,
                 'target_distance': srcs_dis_value, # [b, 1]
                 }
        return datas

    def _resume_checkpoint(self):
        checkpoint_files = sorted(glob(os.path.join(self.checkpoint_path, '*.tar')))
        if checkpoint_files:
            if self.resume_step != 0:
                latest_checkpoints = os.path.join(self.checkpoint_path, self.resume_step)
            else:
                latest_checkpoints = checkpoint_files[-1]
        
            map_location = self.device
            checkpoint = torch.load(latest_checkpoints, map_location=map_location)
            model_weights = checkpoint['model']
            self.model.load_state_dict(model_weights)
            
            self.start_epoch = checkpoint['epoch'] + 1
            self.optimizer.load_state_dict(checkpoint['optimizer'])

            if self.trainer_config['resume_lr'] != 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.trainer_config['resume_lr']
        else:
            print(f"No checkpoint found in {self.checkpoint_path}, starting from scratch.")

    def _set_train_mode(self):
        self.model.train()

    def _set_eval_mode(self):
        self.model.eval()

    def _save_checkpoint(self, epoch, valid_loss, is_best=False):
        state_dict = {'epoch': epoch,
                      'optimizer': self.optimizer.state_dict(),
                      'model': self.model.state_dict(),
                      'loss': valid_loss}
        if not is_best:
            torch.save(state_dict, os.path.join(self.checkpoint_path, f'model_{str(epoch).zfill(4)}_loss_{valid_loss:.4f}.tar'))
        if is_best:
            # delect previous best
            for file in os.listdir(self.checkpoint_path):
                if file.startswith("best_epoch") and file.endswith(".tar"):
                    os.remove(os.path.join(self.checkpoint_path, file))
            best_model_filename = f'best_epoch_{str(epoch).zfill(4)}_loss_{valid_loss:.4f}.tar'
            print(f"Validation loss improved to {valid_loss}. Saving best model as {best_model_filename}...")
            torch.save(state_dict, os.path.join(self.checkpoint_path, best_model_filename))
        
