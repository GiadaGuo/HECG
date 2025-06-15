from torch.utils.data import DataLoader
# import sys
import torch
import os
import numpy as np
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from HECG.hecg_model_utils import get_vit200,eval_transforms

def load_batch200(folder_path,batch_size=5):
    tensor_list = []

    for fname in sorted(os.listdir(folder_path)):
        if fname.endswith('.npy'):
            s_path = os.path.join(folder_path, fname)
            sample = np.load(s_path)
            x = eval_transforms(sample).unsqueeze(0)  # [1, 12, 1, 200]
            tensor_list.append(x)

    all_series = torch.vstack(tensor_list)  # [N, 12, 1, 200]-> e.g.[124975, 12, 1, 200]

    # 按 batch_size 沿第一维度分成批次（去掉不能整除的部分），存放在列表中
    # 例：维度(124975, 12, 1, 200)张量=>[(5,12, 1, 200),(5,12, 1, 200),...(5,12, 1, 200)]共24995个
    batch_200 = [all_series[i:i+batch_size] for i in range(0, all_series.size(0)-all_series.size(0)%batch_size, batch_size)]

    return batch_200

def extract_x200(folder_path=None,batch_size=5,device200=None,output_dir=None,model200=None):

    batch_200 = load_batch200(folder_path)

    count_id = 0

    for batch_idx, batch in enumerate(batch_200):  # 每个batch是(5,12, 1, 200)的tensor
        batch = batch.to(device200)
        with torch.no_grad():
            features = model200(batch)  # [bs, 384]，一直在GPU

        features_200_batch = features.cpu()  # 拼接成一个矩阵


        if batch_idx % 5 == 0:
            count_id += 1

        batch_file_path = os.path.join(output_dir, f'HR{count_id:05d}_x200_batch_{batch_idx + 1}.pth')
        torch.save(features_200_batch, batch_file_path)


if __name__ == '__main__':
    folder_path = 'data/ecg_test1_200'
    model200_path = 'test1/ckpts/pretrain_vit1/checkpoint.pth'
    output_dir = 'data/ecg_test2_f200'

    device200 = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model200 = get_vit200(pretrained_weights=model200_path).to(device200)

    extract_x200(folder_path=folder_path,batch_size=5,device200=device200,output_dir=output_dir,model200=model200)






