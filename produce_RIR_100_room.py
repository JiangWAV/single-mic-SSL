import os
import json
import soundfile as sf
import numpy as np
from tqdm import tqdm
from FRAM_RIR import FRAM_RIR
import torch
import random

seed = 43
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

def sample_src_pos(room_dim, rt, edge, mic_posi, dis_range, num_per_mic):
    # random sample the source positon
    src_pos = []
    dis_3ds = []
    dis_2ds = []
    
    while len(src_pos) < num_per_mic:
        pos = np.random.uniform(np.array([edge, edge, 0.5]), np.array([room_dim[0]-edge, room_dim[1]-edge, 2.0]))
        dis_3d = np.linalg.norm(pos - mic_posi)
        dis_2d = np.linalg.norm(pos[:2] - mic_posi[:2])
        if dis_2d >= dis_range[0] and dis_2d <= dis_range[1]:
            src_pos.append(pos)
            dis_3ds.append(dis_3d)
            dis_2ds.append(dis_2d)
        
    return np.stack(src_pos, 0), dis_3ds, dis_2ds

def split_dataset(data):
    train_dataset = []
    valid_dataset = []
    test_dataset  = []
    for sub_data in data:
        random.shuffle(sub_data)
        
        n_total = len(sub_data)
        n_train = int(n_total * 0.8)
        n_valid = int(n_total * 0.05)
        
        train_dataset.append(sub_data[:n_train])
        valid_dataset.append(sub_data[n_train:n_train+n_valid])
        test_dataset.append (sub_data[n_train+n_valid:])
    return train_dataset,valid_dataset,test_dataset

def generate_dataset(sr, room_dims, total_rts, edge, total_mic_posis, root_path, dis_range, num_per_mic = 50):  
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    save_name = 'roomFRAM'  + '_room_' + str(len(total_mic_posis)) + '_per_' + str(num_per_mic)
    room_path = os.path.join(root_path, 'data')
    if not os.path.exists(room_path):
        os.makedirs(room_path)
    
    room_num = len(room_dims)
    
    # store room information
    total_room_info = []

    # each room informations
    for room_i in tqdm(range(room_num), ncols=50):
        room_dim = room_dims[room_i] # [x, y, z]
        rt60 = total_rts[room_i]

        # store one mic data
        oneroom_totalmic_info = []
        # mic position
        mic_posis = total_mic_posis[room_i]                        # [10000, 3], [x, y, z]
        for i_mic in tqdm(range(len(mic_posis)), ncols=50):
            mic_posi = mic_posis[i_mic]
            src_pos, dis_3ds, dis_2ds = sample_src_pos(room_dim, rt, edge, mic_posi, dis_range, num_per_mic)
            
            array_pos = np.expand_dims(mic_posi, axis=0)
            mic_posi = np.expand_dims(mic_posi, axis=0)

            # generate RIR (Simulate the collected sounds)
            rir, rir_direct = FRAM_RIR(mic_posi, sr, rt60, room_dim, src_pos, array_pos)
            rir = rir[0, :, :]
            rir_direct = rir_direct[0, :, :]
            
            oneroom_onemic_info = []
            for i_src in range(rir.shape[0]):
                # Reverberant sound
                file_name = save_name + "room" + "{:05d}".format(room_i)  + "_imic_" + "{:05d}".format(i_mic) + "_isrc_" + "{:05d}".format(i_src) + ".wav"
                save_path = os.path.join(room_path, file_name)
                sf.write(save_path, rir[i_src], sr, format='WAV')

                # Direct sound
                file_name_2 = save_name + "room" + "{:05d}".format(room_i) + "_imic_" + "{:05d}".format(i_mic) + "_isrc_" + "{:05d}".format(i_src) + "direct" + ".wav"
                save_path_2 = os.path.join(room_path, file_name_2)
                sf.write(save_path_2, rir_direct[i_src], sr, format='WAV')

                info = {
                        "room_index": room_i,
                        "mic_index": i_mic,
                        "name": file_name,
                        "rever_path": save_path,
                        "direct_path": save_path_2,
                        "rir_len": len(rir[i_src]),
                        "room_dim": room_dim, 
                        "sample rate": sr,
                        "rt60": rt60,
                        "source_posi": src_pos[i_src].tolist(),
                        "mic_posi": mic_posi[0].tolist(), 
                        "dis_to_mic_3D": dis_3ds[i_src].tolist(),
                        "dis_to_mic_2D": dis_2ds[i_src].tolist()
                        }
                
                oneroom_onemic_info.append(info)
            oneroom_totalmic_info.append(oneroom_onemic_info)
        total_room_info.append(oneroom_totalmic_info)

    # save datasets
    if not os.path.exists(os.path.join(root_path, 'data_info')):
        os.makedirs(os.path.join(root_path, 'data_info'))
    
    train_dataset,valid_dataset,test_dataset = split_dataset(total_room_info)

    base_dir   = root_path + '/' + 'data_info' + '/'
    total_file = os.path.join(base_dir, 'total_data_info.json')
    train_file = os.path.join(base_dir, 'train_info.json')
    valid_file = os.path.join(base_dir, 'val_info.json')
    test_file  = os.path.join(base_dir, 'test_info.json')
    
    with open(total_file, 'w') as f:
        json.dump(total_room_info, f, indent=4)
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_dataset, f, ensure_ascii=False, indent=4)
    with open(valid_file, 'w', encoding='utf-8') as f:
        json.dump(valid_dataset, f, ensure_ascii=False, indent=4)
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_dataset, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    # sample rate 
    sr = 16000
    
    # room dim setting
    num_rooms = 100                          
    center_dims = np.array([5.9, 6.9, 2.9])  # Dimensions of the room
    delta_dims = 0.5                         # random size variation ±0.5

    # The size of 100 rooms
    room_dims = [list(center_dims + np.random.uniform(-delta_dims, delta_dims, size=3))
                for _ in range(num_rooms)]

    # rt60 (Reverberation Time) setting
    total_rts = []
    for i in range(len(room_dims)):
        rt = 0.6 + np.random.uniform(-0.2, 0.15)
        total_rts.append(rt)
        
    # microphone positions
    total_mic_posis = []
    edge = 0.5                                #The distance to the wall
    for room_dim in room_dims:
        edges = [edge] * 3
        room_dim_noedge = [i - edge for i in room_dim ]

        # 10*10 microphone positions
        x_mic_posis = np.linspace(edges[0], room_dim_noedge[0], 10)     
        y_mic_posis = np.linspace(edges[1], room_dim_noedge[1], 10)
        mic_z = 1.5
        mic_xy_posis = np.array(np.meshgrid(x_mic_posis, y_mic_posis)).T.reshape(-1, 2)
        mic_xyz_posis = np.column_stack((mic_xy_posis, np.full(mic_xy_posis.shape[0], mic_z)))

        total_mic_posis.append(mic_xyz_posis)

    # distance range 
    dis_range = [0.1, 6]
    
    # generate datasets
    root_path = os.getcwd()+'/dataset/FramRIRs/RIR_100room_100mic_50src_0224'
    generate_dataset(sr, room_dims, total_rts, edge, total_mic_posis, root_path, dis_range, num_per_mic = 50)
    
