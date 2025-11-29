import os
import toml
import random
import torch
import numpy as np
import soundfile as sf
from torch.utils import data


class MyDataset(data.Dataset):
    def __init__(self,
                n_fft=512, 
                hop_length=128, 
                win_length=512,
                **kwargs):
        super().__init__()

        random.seed(7)

        # fft
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        self.dataset_size = kwargs.get('dataset_size')

        # real data dirs
        audio_root_dirs   = kwargs.get('audio_root_dirs')
        mic_pos_root_dirs = kwargs.get('mic_pos_root_dirs')
        

        used_file_index = kwargs.get('used_experiment_name')
        interval = 0.1           # The recording interval is taken once every 100ms
        used_internal = 0.2      # An interval of 200ms used

        samplerate = 16000

        # total used audio & mic_pos
        self.total_used_audio = []
        self.total_posi = []
        self.total_src_posi = []
        for index in used_file_index:
            audio_path = os.path.join(audio_root_dirs, f"recorded_audio{index}.wav")
            posi_path = os.path.join(mic_pos_root_dirs, f"mic_pos_{index}.txt")
            src_posi = self.return_src_posi(index)

            # read audio
            audio, samplerate = sf.read(audio_path) # [wave length, channel]
            if audio.ndim > 1:
                audio = audio[:, 0]
            else:
                audio = audio
            # read path 
            posis = self.read_txt_lines(posi_path)        # [len posi, 2] ([x, y])

            # slice
            audio_slicses = []
            audio_step    = int(used_internal * samplerate)
            audio_used_num = len(audio) // audio_step      # calculate the number of audio that can be used within used_internal
            for i in range(0, audio_used_num):
                audio_slicses.append(audio[i * audio_step:i * audio_step + audio_step])

            posi_slicses = []
            posi_step = int(used_internal / interval) 
            posi_used_num = len(posis) // posi_step        # calculate the number of positions that can be used within used_internal

            for i in range(0, posi_used_num):
                # The average of adjacent points
                x_point = np.mean(posis[i*posi_step:i*posi_step+posi_step, 0])
                y_point = np.mean(posis[i*posi_step:i*posi_step+posi_step, 1])
                posi_slicses.append(np.array([x_point, y_point]))
            print(f"index: {index}, audio_slice: {len(audio_slicses)}, posi_slice: {len(posi_slicses)}")

            assert len(audio_slicses) == len(posi_slicses), \
                f"The number of audio slices({len(audio_slicses)})and the quantity of location data({len(posi_slicses)}) are mismatching"

            # total audio
            self.total_used_audio.append(audio_slicses)
            self.total_posi.append(posi_slicses)
            self.total_src_posi.append(src_posi)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        # chosen experiment sample
        ran_ex_index = random.randint(0, len(self.total_used_audio) - 1)
        audio_slice = self.total_used_audio[ran_ex_index]
        posi_slice = self.total_posi[ran_ex_index]
        src_posi = self.total_src_posi[ran_ex_index] # np, size [2, ]

        # chosen audio sample
        ran_sample = random.randint(0, len(audio_slice) - 1)
        audio_sample = audio_slice[ran_sample]       # np, size [wave length, ]
        posi_sample = posi_slice[ran_sample]         # np, size [2, ]

        # cal dis
        dis = np.sqrt((src_posi[0] - posi_sample[0])**2 + (src_posi[1] - posi_sample[1])**2)

        # infos
        info = {
            "index": index,
            "ran_ex_index": ran_ex_index,
            "ran_sample": ran_sample,
            "audio_length": len(audio_sample),
            "dis": dis,
            "src_posi": src_posi,
            "posi_sample": posi_sample
        }

        # to tensor
        audio_sample = torch.tensor(audio_sample, dtype=torch.float32)
        dis = torch.tensor(dis, dtype=torch.float32)

        return  audio_sample, dis, info
            

    def return_src_posi(self, index):
        if index in [1, 2, 3, 4, 5]:
            src_posi = np.array([0, 3.0])
        elif index in [6, 7, 8, 9, 10]:
            src_posi = np.array([2.5, 3.0])
        elif index in [11, 12, 13, 14, 15]:
            src_posi = np.array([0.5, 3.0])
        elif index in [16, 17, 18, 19, 20]:
            src_posi = np.array([1, 3.0])
        elif index in [21, 22, 23, 24, 25]:
            src_posi = np.array([1.5, 3.0])
        elif index in [26, 27, 28, 29, 30]:
            src_posi = np.array([2.0, 3.0])
        elif index == 31:
            src_posi = np.array([2.0, 2.5])
        elif index == 32:
            src_posi = np.array([2.5, 2.5])
        elif index == 33:
            src_posi = np.array([1.5, 2.5])
        elif index == 34:
            src_posi = np.array([2.0, 2.0])
        elif index == 35:
            src_posi = np.array([2.5, 2.0])
        elif index == 36:
            src_posi = np.array([2.5, 1.5])
        elif index == 37:
            src_posi = np.array([2.5, 1.5])
        return src_posi

    def read_txt_lines(self, txt_path):
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        lines = [line.rstrip('\n') for line in lines]
        out = np.array([list(map(float, s.split())) for s in lines])
        return out


if __name__ == "__main__":

    config = toml.load('D:\work_code\distance_esti\configs\simulation\config.toml')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_dataset = MyDataset(**config['FFT'], **config['train_gene_setting'])
    train_dataloader = data.DataLoader(train_dataset, **config['train_dataloader'])
    
    from tqdm import tqdm  
    for i, (data) in tqdm(enumerate(train_dataloader)):
        print(data)