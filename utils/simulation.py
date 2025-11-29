import json
import random
import torch
import numpy as np
import soundfile as sf
from scipy.signal import convolve
from torch.utils import data


class MyDataset(data.Dataset):
    def __init__(self, 
                index_file, 
                shuffle, 
                num_tot, 
                wav_len=0, 
                n_fft=512, 
                hop_length=256, 
                win_length=512,
                **kwargs):
        super().__init__()
        
        random.seed(7)
        
        # traing phase epoch
        self.phase_epoch = 300
        self.phase_notice = True
        self.test_mode = False
        
        # configs
        self.sample_rate = kwargs.get('sample_rate') 
        self.input_len   = wav_len                                          # s input sample length
        self.wav_len_num = int(self.input_len * self.sample_rate)           # 16k

        # fft
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        
        # generation configs
        self.dataset_size = kwargs.get('dataset_size')     
        self.n_src        = kwargs.get('n_src')               
        
        # mix setting 
        self.sample_random = kwargs.get('random_mix_posi')  # random mix position
        self.tgt_amplify   = kwargs.get('tgt_amplify')
        self.amplify_range = kwargs.get('amplify_range')

        
        # Load Json files
        RIR_index_file, audio_index_file = index_file[0], index_file[1]
        self.used_speaker, self.RIR_room_lyr1 = self._process_RIR_Audio(RIR_index_file, audio_index_file)
        
        # actual room num
        self.total_room_num = len(self.RIR_room_lyr1)
        
        # training mode
        self.train_mode = 'full_random'
        
        # room / mic / RIR information
        used_room_index = range(self.total_room_num)
        room_RIR_info = [self.RIR_room_lyr1[i] for i in used_room_index]
        self.room_mic_RIR_info = []
        for mic_list in room_RIR_info:
            used_mic_index = range(len(mic_list))
            self.room_mic_RIR_info.extend([mic_list[i] for i in used_mic_index])
        self.mic_num = len(self.room_mic_RIR_info)
        self.total_RIR_num = self.count_elements(self.room_mic_RIR_info)
                        
        print(f"Total room data: {len(used_room_index)}")
        print(f"Total mic num  : {self.mic_num}")
        print(f"Total RIR data : {self.total_RIR_num}")
        
    def find_effective_start(self, signal, threshold=0.01):
        # the energy of the signal
        energy = signal ** 2
        # normalization
        normalized_energy = energy / np.max(energy)

        start_index = np.where(normalized_energy > threshold)[0][0]
        return start_index 
    
    def set_epoch(self, epoch):
        # current epoch
        self.current_epoch = epoch
        
        if self.current_epoch == self.phase_epoch:
            print('change training method, epoch:', self.current_epoch)
    
    def set_test(self):
        self.test_mode = True
        print('set test mode', self.test_mode)
    
    def __getitem__(self, idx): 
        # RIR for srcs
        mic_index = random.sample(range(self.mic_num), 1)[0]
        used_mic = self.room_mic_RIR_info[mic_index]
            
        RIR_files = []
        RIR_diss = []

        # randomly select self.n_src unique index
        candidates = random.sample(range(len(used_mic)), self.n_src)

        for candidate in candidates:
            used_sample_info = used_mic[candidate]
            RIR_diss.append(round(used_sample_info['dis_to_mic_2D'], 5)) # 2
            
            # load audio files
            audio, fs = sf.read(used_sample_info['rever_path'], dtype="float32")
            RIR_files.append(audio)
                    
        # sorted RIR_diss
        sorted_indices = np.argsort(RIR_diss)
        
        RIR_diss = [RIR_diss[i] for i in sorted_indices]
        RIR_files = [RIR_files[i] for i in sorted_indices]
        
        srcs_dis_value = RIR_diss
        
        # distance information
        dis_info = {
            'srcs_dis_value': srcs_dis_value
        }
        
        clean_speechs    = []
        path = self.used_speaker[0][0]['speech_path']
        audio, fs = sf.read(path, dtype="float32")
        clean_speechs.append(audio)
            
        re_speechs               = []
        clean_direct_enh_speechs = []
        
        for RIR, speech in zip(RIR_files, clean_speechs):     
            if self.tgt_amplify:
                # the RMS value of the current sound segment
                current_rms = np.sqrt(np.mean(speech**2))
                current_rms_db = 20 * np.log10(current_rms)
                
                if isinstance(self.amplify_range, list): 
                    tgt_db = round(random.uniform(*self.amplify_range), 3)
                else: 
                    tgt_db = round(self.amplify_range, 3)
                
                db_difference = tgt_db - current_rms_db
                amplify_ratio = 10 ** (db_difference / 20)
                speech = speech * amplify_ratio

            # RIR start point
            start_index = self.find_effective_start(RIR)
            
            # for convolve
            effective_RIR = RIR[:]
            
            # RIR filter
            re_speech = convolve(speech, effective_RIR, mode = 'full', method = 'fft')                    # 'auto' or 'fft'
            re_speechs.append(re_speech)
            
            # for early & direct rir
            early_start_index = max(0, start_index - int(0.006 * self.sample_rate))
            early_end_index = min(len(RIR), start_index + int(0.050 * self.sample_rate))

            early_reflection = np.zeros_like(RIR)
            early_reflection[early_start_index:early_end_index] = RIR[early_start_index:early_end_index]
            
            # early & direct  
            direct_speech = convolve(speech, early_reflection, mode = 'full', method = 'fft')               # 'auto' or 'fft'
            clean_direct_enh_speechs.append(direct_speech)
        
        target_reverb= self._adjust_audio(re_speechs, clean_direct_enh_speechs, self.n_src, self.wav_len_num)
        
        return target_reverb, dis_info
   
    @staticmethod
    def collate_fn(batch):
        # target: separated reverb speech
        tensor_list1 = [torch.tensor(data[0], dtype=torch.float32) for data in batch]
        reverb_batch = torch.nn.utils.rnn.pad_sequence(tensor_list1, batch_first=True)
        
        # RIR's distance
        srcs_dis_value_batch = torch.stack([torch.tensor(data[1]['srcs_dis_value'], dtype=torch.float32) for data in batch])
        RIR_dis_batch = {'srcs_dis_value':srcs_dis_value_batch}        
        return reverb_batch, RIR_dis_batch
    
    def __len__(self):
        return self.dataset_size

    def _adjust_audio(self, input_group, target_group, src_num, wav_len_num):
        input_len_list  = [len(item) for item in input_group]
        target_len_list = [len(item) for item in target_group]
        
        target_reverb = np.stack([np.zeros(wav_len_num, dtype=np.float32) for i in range(src_num)], axis=0)

        for i in range(src_num):    
            input_len  = input_len_list[i]
            target_len = target_len_list[i]
            
            # Determine the starting point for cutting or filling
            if input_len >= wav_len_num and target_len >= wav_len_num:
                if self.sample_random:
                # If the lengths of both are greater than or equal to wav_len_num, randomly select the starting point
                    start = random.randint(0, min(input_len, target_len) - wav_len_num)
                else:
                    start = (min(input_len, target_len) - wav_len_num) // 2
                    
                input_cut = input_group[i][start:start + wav_len_num]
            else:
                # If any length is less than wav_len_num, calculate the random fill length on the left side
                cut_length = min(target_len, wav_len_num)
                max_left_padding = wav_len_num - cut_length
                if self.sample_random:
                    left_padding = random.randint(0, max_left_padding) if max_left_padding > 0 else 0
                else:
                    left_padding = max_left_padding // 2
                right_padding = wav_len_num - cut_length - left_padding
                
                # Start from 0 and extract as much as possible. Fill in 0 on the left and right sides as needed.
                input_cut = np.pad(input_group[i][:cut_length], (left_padding, right_padding), 'constant')
                start = left_padding
            assert len(input_cut) == wav_len_num
            target_reverb[i] = input_cut
        return target_reverb
                
    def count_elements(self, nested_list):
        count = 0
        for element in nested_list:
            if isinstance(element, list):
                count += self.count_elements(element)
            else:
                count += 1
        return count
    
    def _process_RIR_Audio(self, RIR_path, audio_path):
        # audio path: the oringinal chirp audio
        # RIR path  : the reverbrant record audio
        audio_info_list  = []        
        with open(RIR_path, 'r') as f:
            RIR_index = json.load(f)

        with open(audio_path, 'r') as f:
            audio_index = json.load(f)
            
        for spker, audio_info in zip(audio_index.keys(), audio_index.values()):
            one_spker = []
            for i in range(len(audio_info)):    
                one_spker.append(audio_info[i])
            audio_info_list.append(one_spker)
        return audio_info_list, RIR_index