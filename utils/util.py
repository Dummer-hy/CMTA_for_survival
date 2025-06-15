import os
import random
import numpy as np

import torch
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler

def collate_MIL_survival(batch):
    img = torch.cat([item[0] for item in batch], dim = 0)
    omic = torch.cat([item[1] for item in batch], dim = 0).type(torch.FloatTensor)
    label = torch.LongTensor([item[2] for item in batch])
    event_time = np.array([item[3] for item in batch])
    c = torch.FloatTensor([item[4] for item in batch])
    return [img, omic, label, event_time, c]

def collate_MIL_survival_cluster(batch):
    img = torch.cat([item[0] for item in batch], dim = 0)
    cluster_ids = torch.cat([item[1] for item in batch], dim = 0).type(torch.LongTensor)
    omic = torch.cat([item[2] for item in batch], dim = 0).type(torch.FloatTensor)
    label = torch.LongTensor([item[3] for item in batch])
    event_time = np.array([item[4] for item in batch])
    c = torch.FloatTensor([item[5] for item in batch])
    return [img, cluster_ids, omic, label, event_time, c]

def collate_MIL_survival_sig(batch):
    img = torch.cat([item[0] for item in batch], dim = 0)
    omic1 = torch.cat([item[1] for item in batch], dim = 0).type(torch.FloatTensor)
    omic2 = torch.cat([item[2] for item in batch], dim = 0).type(torch.FloatTensor)
    omic3 = torch.cat([item[3] for item in batch], dim = 0).type(torch.FloatTensor)
    omic4 = torch.cat([item[4] for item in batch], dim = 0).type(torch.FloatTensor)
    omic5 = torch.cat([item[5] for item in batch], dim = 0).type(torch.FloatTensor)
    omic6 = torch.cat([item[6] for item in batch], dim = 0).type(torch.FloatTensor)

    label = torch.LongTensor([item[7] for item in batch])
    event_time = np.array([item[8] for item in batch])
    c = torch.FloatTensor([item[9] for item in batch])
    return [img, omic1, omic2, omic3, omic4, omic5, omic6, label, event_time, c]


def make_weights_for_balanced_classes_split(dataset):
    N = float(len(dataset))  # 总样本数
    weight_per_class = []

    # 为每个类别计算权重，避免除以零
    for c in range(len(dataset.slide_cls_ids)):
        class_size = len(dataset.slide_cls_ids[c])
        if class_size == 0:
            # 如果某个类没有样本，可以为它分配一个默认的权重（如1）
            print(f"Warning: Class {c} has no samples.")
            weight_per_class.append(0)  # 可以选择其他合理的默认值，例如 1
        else:
            weight_per_class.append(N / class_size)  # 计算权重

    weight = [0] * int(N)

    # 为每个样本分配权重
    for idx in range(len(dataset)):
        y = dataset.getlabel(idx)  # 获取样本的标签
        weight[idx] = weight_per_class[y]  # 给该样本赋权重

    return torch.DoubleTensor(weight)


class SubsetSequentialSampler(Sampler):
    """Samples elements sequentially from a given list of indices, without replacement.
    Arguments:
        indices (sequence): a sequence of indices
    """
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

def get_split_loader(split_dataset, training = False, testing = False, weighted = False, modal='coattn', batch_size=1):
    """
        return either the validation loader or training loader 
    """
    if modal == 'coattn':
        collate = collate_MIL_survival_sig
    elif modal == 'cluster':
        collate = collate_MIL_survival_cluster
    else:
        collate = collate_MIL_survival

    kwargs = {'num_workers': 0} if torch.cuda.is_available() else {}
    if not testing:
        if training:
            if weighted:
                weights = make_weights_for_balanced_classes_split(split_dataset)
                loader = DataLoader(split_dataset, batch_size=batch_size, sampler = WeightedRandomSampler(weights, len(weights)), collate_fn = collate, **kwargs)    
            else:
                loader = DataLoader(split_dataset, batch_size=batch_size, sampler = RandomSampler(split_dataset), collate_fn = collate, **kwargs)
        else:
            loader = DataLoader(split_dataset, batch_size=batch_size, sampler = SequentialSampler(split_dataset), collate_fn = collate, **kwargs)
    
    else:
        ids = np.random.choice(np.arange(len(split_dataset), int(len(split_dataset)*0.1)), replace = False)
        loader = DataLoader(split_dataset, batch_size=1, sampler = SubsetSequentialSampler(ids), collate_fn = collate, **kwargs )

    return loader

def set_seed(seed=7):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True