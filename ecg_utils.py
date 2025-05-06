import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class NumpyDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): 包含所有 .npy 文件的目录
            transform (callable, optional): 对每个样本应用的变换（如转为 tensor 等）

        使用方法：
            dataset = NumpyDataset(root_dir=root_dir, transform=transform)
        """
        self.root_dir = root_dir
        self.transform = transform

        # 获取所有 .npy 文件路径
        self.file_list = [f for f in os.listdir(root_dir) if f.endswith('.npy')]
        self.file_list.sort()  # 可选，保证顺序一致

    def __len__(self):
        return len(self.file_list)

    #定义迭代器如何获取数据集中的单个样本
    #DataLoader 会自动调用 __getitem__ 方法来获取单个样本，并将它们组合成批量数据
    def __getitem__(self, idx):
        file_path = os.path.join(self.root_dir, self.file_list[idx])
        sample = np.load(file_path)

        if self.transform is not None:
            #transform是 ECGDataAugmentationDINO的实例
            #调用transform则自动运行了__call__，为每个sample生成了crops
            sample = self.transform(sample)

        return sample #根据1个sample生成的列表crops

class MultiCropWrapper(nn.Module):
    """
    用于一次性处理同分辨率图像：

    对应于单一分辨率的输入会被组合在一起，并且在相同分辨率的输入上运行一次前向传播。
    执行多次前向传播，次数等于使用的不同分辨率的数量
    将所有输出特征连接起来，并在这些连接后的特征上运行头部（head）的前向传播

    实例化方法：
    teacher = ecg_utils.MultiCropWrapper(
        teacher, #backbone
        DINOHead(embed_dim, args.out_dim, args.use_bn_in_head), #head
    )

    传入数据：
    teacher:images[:2]=>list [(64,12,1,160),(64,12,1,160)]
    student: images=>list [(64,12,1,160),(64,12,1,160),(64,12,1,80),...]
    """
    def __init__(self, backbone, head): #backbone实例化时填student或teacher
        super(MultiCropWrapper, self).__init__()
        # disable（用nn.Identity()完成） layers dedicated to ImageNet labels classification
        # nn.Identity 是一个通过输入直接产生输出的层，没有任何计算；
        # 只关心模型的特征提取部分，跳过模型的全连接层和头部计算（拼接后统一进行）
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        # convert to list
        if not isinstance(x, list):
            x = [x]

        #3）torch.cumsum(..., 0):沿第一个维度（即行）计算累积和
        # idx_crops示例：torch.tensor([2])
        idx_crops = torch.cumsum(
            #2)用于找到张量中连续的唯一值，并可选地返回每个唯一值的计数
            #返回一个元组，其中第一个元素是唯一值，第二个元素是每个唯一值的计数（torch.tensor([160]),torch.tensor([2])）
            #torch.unique_consecutive(..., return_counts=True)[1] 表示提取计数
            #从tensor(160,160)提取出160的计数为torch.tensor([2])
            torch.unique_consecutive(
            #1)提取x中元素最后一个维度，转换为一个 PyTorch 张量
            #每个global crop的维度是(12,1,160)，local crop的维度是(12,1,80)
            #teacher: [(64,12,1,160),(64,12,1,160)] => [160,160]=>torch.tensor([160, 160])
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True, #返回每个唯一值的计数
        )[1], 0)

        #初始化两个变量：start_idx 和 output
        #output 用于存储拼接后的特征，初始为空张量，后续逐步拼接
        #torch.empty(0)：生成空张量
        #.to(x[0].device) 将空张量移动到与 x[0] 相同的设备上（例如 CPU 或 GPU）
        # 这确保了后续操作中，output 和 x 中的张量在同一个设备上，避免设备不匹配的错误。
        start_idx, output = 0, torch.empty(0).to(x[0].device)

        #利用idx_crops提供的累积索引
        #分段（start_idx: end_idx）提取相同分辨率的crops
        #示例idx_crops=torch.tensor([2])，则end_idx遍历idx_crops
        for end_idx in idx_crops: #for循环可用于遍历一维张量中元素，end_idx 是一个标量张量不是Python基本数据类型

            #固定分辨率得到对应特征
            #torch.cat(x[start_idx: end_idx])：默认拼接第0维
            #拼接(64, 12, 1, 160)和(64, 12, 1, 160)=>(128, 12, 1, 160)
            #拼接完成输入self.backbone进行前向计算！所以global和local是分开计算的
            #self.backbone就是VIT！VIT做了Transform流程里的Norm为止，后续还要MLP和LN（由DINO HEAD完成）
            #_out是MLP和LN处理前的VIT特征，但拼接前分别是global和local的
            _out = self.backbone(torch.cat(x[start_idx: end_idx])) #输入维度：(128, 12, 1, 160)

            # 如果是XCit的元组输出则取第一个元素作为特征；自己的VIT可以直接拼接
            if isinstance(_out, tuple):
                _out = _out[0]

            # accumulate outputs：将所有输出特征连接起来
            # output初始化是空张量；
            # output按顺序拼接global views和local views，循环结束使得所有global feature在前，local feature在后
            output = torch.cat((output, _out)) #默认按第0维拼接
            start_idx = end_idx

        # Run the head forward on the concatenated features.
        return self.head(output)






