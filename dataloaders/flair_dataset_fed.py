import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle
from pdb import set_trace as stop
from dataloaders.data_utils import get_unk_mask_indices,image_loader

import h5py


class FlairFedDataset(Dataset):
    def __init__(self, inp_data, split, num_labels, data_file, img_root, curr_user=None, max_samples=-1,transform=None,known_labels=0,testing=False, label_mapping=None, fine_grained_label_mapping=None):
        super(FlairFedDataset, self).__init__()
        # print(data_file)
        #self.split_data = h5py.File('/home/liujack/multi_label/C-Tran/data/flair/cent_data.hdf5', 'r')
        self.split_data = inp_data
        
        self.split = split
       
        self.fine_grained_label_mapping = fine_grained_label_mapping
        self.label_mapping = label_mapping
        if max_samples != -1:
            self.split_data = self.split_data[0:max_samples]
        self.img_root = img_root
        self.transform = transform
        self.num_labels = num_labels
        self.known_labels = known_labels
        self.testing = testing
        self.curr_user = curr_user
        self.image_id_list = list(self.split_data[self.split][self.curr_user]['image_ids'])
        self.image_list = list(self.split_data[self.split][self.curr_user]['images'])
        self.label_list = list(self.split_data[self.split][self.curr_user]['labels'])
        self.fg_label_list = list(self.split_data[self.split][self.curr_user]['fine_grained_labels'])

    def __len__(self):
        return len(self.image_id_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # take a sample
        image_ID = self.image_id_list[idx]
        # img = np.array(self.split_data[self.split][self.curr_user][image_ID]['image'])
        img = self.image_list[idx]
        image = self.transform(img)

        if self.fine_grained_label_mapping != None:
            # fine grained labels are used
            # labels_str = np.array(self.split_data[self.split][image_ID]['fine_grained_labels'])
            labels_str = self.fg_label_list[idx]
        else:
            # coarse grained labels are used
            # labels_str = np.array(self.split_data[self.split][image_ID]['labels'])
            labels_str = self.label_list[idx]
            assert self.label_mapping != None

        # fg_labels = np.array(self.split_data[self.split][image_ID]['fine_grained_labels'])
        # image_ID = self.split_data[idx]['file_name']
        # img_name = os.path.join(self.img_root,image_ID + '.jpg')
        # image = image_loader(img_name,self.transform)
        labels_str = labels_str.tolist()
        labels_str = str(labels_str)[2:-1].split('|')
        
        tran_labels = [0] * self.num_labels
        if self.fine_grained_label_mapping != None:
            for label in labels_str:
                tran_labels = list(map(lambda x, y: x | y, tran_labels, self.fine_grained_label_mapping[label]))
        else:
            for label in labels_str:
                tran_labels = list(map(lambda x, y: x | y, tran_labels, self.label_mapping[label]))
        assert tran_labels.count(1) == len(labels_str)
        labels = torch.Tensor(tran_labels)

        unk_mask_indices = get_unk_mask_indices(image,self.testing,self.num_labels,self.known_labels)
        
        mask = labels.clone()
        # perform the random masking 25%
        mask.scatter_(0,torch.Tensor(unk_mask_indices).long() , -1)
        sample = {}
        sample['image'] = image 
        sample['labels'] = labels
        sample['mask'] = mask
        sample['imageIDs'] = image_ID
        return sample


