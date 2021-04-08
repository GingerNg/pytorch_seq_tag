import torch
from torch.utils.data.dataset import Dataset
import numpy as np
from torch.utils.data import DataLoader

POSE_VOCABULARY = ["O", "B-KEY", "I-KEY", "E-KEY"]

class CustomDataSet(Dataset):
    def __init__(
            self,
            poses,
            texts,
            labels):
        self.poses = poses
        self.texts = texts
        self.labels = labels

    def __getitem__(self, index):
        pose = self.poses[index]
        text = self.texts[index]
        label = self.labels[index]
        return pose, text, label

    def __len__(self):
        count = len(self.poses)
        assert len(self.poses) == len(self.labels)
        return count

def get_loader_mock(batch_size):
    # img_train = loadmat(path+"train_img.mat")['train_img']
    # img_test = loadmat(path + "test_img.mat")['test_img']
    # text_train = loadmat(path+"train_txt.mat")['train_txt']
    # text_test = loadmat(path + "test_txt.mat")['test_txt']
    # label_train = loadmat(path+"train_img_lab.mat")['train_img_lab']
    # label_test = loadmat(path + "test_img_lab.mat")['test_img_lab']

    # label_train = ind2vec(label_train).astype(int)
    # label_test = ind2vec(label_test).astype(int)
    frame_size = 50
    train_num = 300
    test_num = 100
    dev_num = 100
    pose_train = torch.rand(train_num, frame_size, 18, 2)  # openpose json --> embedding
    pose_dev = torch.rand(dev_num, frame_size, 18, 2)
    pose_test = torch.rand(test_num, frame_size, 18, 2)

    text_train = torch.rand(train_num, 20, 768)
    text_dev = torch.rand(dev_num, 20, 768)
    text_test = torch.rand(test_num, 20, 768)

    label_train = np.random.randn(train_num, 1)
    label_dev = np.random.randn(dev_num, 1)
    label_test = np.random.randn(test_num, 1)

    poses = {'train': pose_train, 'test': pose_test, "dev": pose_dev}
    texts = {'train': text_train, 'test': text_test, "dev": text_dev}
    labels = {'train': label_train, 'test': label_test, "dev": label_dev}
    dataset = {x: CustomDataSet(poses=poses[x], texts=texts[x], labels=labels[x]) for x in ['train', 'test', "dev"]}

    shuffle = {'train': True, 'test': False, "dev": True}

    dataloader = {x: DataLoader(dataset[x], batch_size=batch_size,
                                shuffle=shuffle[x], num_workers=0) for x in ['train', 'test', "dev"]}

    # img_dim = pose_train.shape[1]
    text_dim = text_train.shape[2]
    num_class = label_train.shape[1]

    input_data_par = {}
    input_data_par['pose_train'] = pose_train
    input_data_par['pose_test'] = pose_test
    input_data_par['pose_dev'] = pose_dev

    input_data_par['text_test'] = text_test
    input_data_par['text_train'] = text_train
    input_data_par['text_dev'] = text_dev

    input_data_par['label_train'] = label_train
    input_data_par['label_test'] = label_test
    input_data_par['label_dev'] = label_dev
    # input_data_par['img_dim'] = img_dim
    input_data_par['text_dim'] = text_dim
    input_data_par['num_class'] = num_class
    return dataloader, input_data_par
