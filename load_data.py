import torch
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os, random
import json
from dataloaders.voc2007_20 import Voc07Dataset
from dataloaders.vg500_dataset import VGDataset
from dataloaders.coco80_dataset import Coco80Dataset
from dataloaders.news500_dataset import NewsDataset
from dataloaders.coco1000_dataset import Coco1000Dataset
from dataloaders.cub312_dataset import CUBDataset

from dataloaders.flair_dataset import FlairDataset
from dataloaders.flair_dataset_fed import FlairFedDataset

import h5py
import warnings
warnings.filterwarnings("ignore")


def get_data(args, curr_user=None):
    dataset = args.dataset
    data_root = args.dataroot
    batch_size = args.batch_size

    rescale = args.scale_size
    random_crop = args.crop_size
    attr_group_dict = args.attr_group_dict
    workers = args.workers
    n_groups = args.n_groups

    normTransform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    scale_size = rescale
    crop_size = random_crop
    if args.test_batch_size == -1:
        args.test_batch_size = batch_size

    trainTransform = transforms.Compose([
                                        transforms.Resize((scale_size, scale_size)),
                                        transforms.Resize((crop_size, crop_size)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normTransform])

    testTransform = transforms.Compose([
                                        transforms.Resize((scale_size, scale_size)),
                                        transforms.CenterCrop(crop_size),
                                        transforms.ToTensor(),
                                        normTransform])

    test_dataset = None
    test_loader = None
    drop_last = False
    if dataset == 'coco':
        coco_root = os.path.join(data_root,'coco')
        ann_dir = os.path.join(coco_root,'annotations_pytorch')
        train_img_root = os.path.join(coco_root,'train2014')
        test_img_root = os.path.join(coco_root,'val2014')
        train_data_name = 'train.data'
        val_data_name = 'val_test.data'
        # Note: the val_test means the validation set and test set are combined
        # 20000 + 20504 = 40504 images
        
        train_dataset = Coco80Dataset(
            split='train',
            num_labels=args.num_labels,
            data_file=os.path.join(coco_root,train_data_name),
            img_root=train_img_root,
            annotation_dir=ann_dir,
            max_samples=args.max_samples,
            transform=trainTransform,
            known_labels=args.train_known_labels,
            testing=False)
        valid_dataset = None
        valid_loader = None
        test_dataset = Coco80Dataset(split='val',
            num_labels=args.num_labels,
            data_file=os.path.join(coco_root,val_data_name),
            img_root=test_img_root,
            annotation_dir=ann_dir,
            max_samples=args.max_samples,
            transform=testTransform,
            known_labels=args.test_known_labels,
            testing=True)
    elif dataset == 'coco1000':
        ann_dir = os.path.join(data_root,'coco','annotations_pytorch')
        data_dir = os.path.join(data_root,'coco')
        train_img_root = os.path.join(data_dir,'train2014')
        test_img_root = os.path.join(data_dir,'val2014')
        
        train_dataset = Coco1000Dataset(ann_dir, data_dir, split = 'train', transform = trainTransform,known_labels=args.train_known_labels,testing=False)
        valid_dataset = Coco1000Dataset(ann_dir, data_dir, split = 'val', transform = testTransform,known_labels=args.test_known_labels,testing=True)
    elif dataset == 'vg':
        vg_root = os.path.join(data_root,'VG')
        train_dir=os.path.join(vg_root,'VG_100K')
        train_list=os.path.join(vg_root,'train_list_500.txt')
        test_dir=os.path.join(vg_root,'VG_100K')
        test_list=os.path.join(vg_root,'test_list_500.txt')
        train_label=os.path.join(vg_root,'vg_category_500_labels_index.json')
        test_label=os.path.join(vg_root,'vg_category_500_labels_index.json')

        train_dataset = VGDataset(
            train_dir,
            train_list,
            trainTransform, 
            train_label,
            known_labels=0,
            testing=False)
        
        valid_dataset = None
        valid_loader = None
        test_dataset = VGDataset(
            test_dir,
            test_list,
            testTransform,
            test_label,
            known_labels=args.test_known_labels,
            testing=True)
    
    elif dataset == 'news':
        drop_last=True
        ann_dir = '/bigtemp/jjl5sw/PartialMLC/data/bbc_data/'

        train_dataset = NewsDataset(ann_dir, split = 'train', transform = trainTransform,known_labels=0,testing=False)
        valid_dataset = NewsDataset(ann_dir, split = 'test', transform = testTransform,known_labels=args.test_known_labels,testing=True)
    
    elif dataset=='voc':
        voc_root = os.path.join(data_root,'voc/VOCdevkit/VOC2007/')
        img_dir = os.path.join(voc_root,'JPEGImages')
        anno_dir = os.path.join(voc_root,'Annotations')
        train_anno_path = os.path.join(voc_root,'ImageSets/Main/trainval.txt')
        test_anno_path = os.path.join(voc_root,'ImageSets/Main/test.txt')

        train_dataset = Voc07Dataset(
            img_dir=img_dir,
            anno_path=train_anno_path,
            image_transform=trainTransform,
            labels_path=anno_dir,
            known_labels=args.train_known_labels,
            testing=False,
            use_difficult=False)
        valid_dataset = None
        valid_loader = None
        # valid_dataset = Voc07Dataset(
        #     img_dir=img_dir,
        #     anno_path=test_anno_path,
        #     image_transform=testTransform,
        #     labels_path=anno_dir,
        #     known_labels=args.test_known_labels,
        #     testing=True)
        test_dataset = Voc07Dataset(
            img_dir=img_dir,
            anno_path=test_anno_path,
            image_transform=testTransform,
            labels_path=anno_dir,
            known_labels=args.test_known_labels,
            testing=True)

    elif dataset == 'cub':
        drop_last=True
        resol=299
        resized_resol = int(resol * 256/224)
        
        trainTransform = transforms.Compose([
            #transforms.Resize((resized_resol, resized_resol)),
            #transforms.RandomSizedCrop(resol),
            transforms.ColorJitter(brightness=32/255, saturation=(0.5, 1.5)),
            transforms.RandomResizedCrop(resol),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), #implicitly divides by 255
            transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
            ])

        testTransform = transforms.Compose([
            #transforms.Resize((resized_resol, resized_resol)),
            transforms.CenterCrop(resol),
            transforms.ToTensor(), #implicitly divides by 255
            transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
            ])
        
        cub_root = os.path.join(data_root,'CUB_200_2011')
        image_dir = os.path.join(cub_root,'images')
        train_list = os.path.join(cub_root,'class_attr_data_10','train_valid.pkl')
        valid_list = os.path.join(cub_root,'class_attr_data_10','train_valid.pkl')
        test_list = os.path.join(cub_root,'class_attr_data_10','test.pkl')

        train_dataset = CUBDataset(image_dir, train_list, trainTransform,known_labels=args.train_known_labels,attr_group_dict=attr_group_dict,testing=False,n_groups=n_groups)
        valid_dataset = CUBDataset(image_dir, valid_list, testTransform,known_labels=args.test_known_labels,attr_group_dict=attr_group_dict,testing=True,n_groups=n_groups)
        test_dataset = CUBDataset(image_dir, test_list, testTransform,known_labels=args.test_known_labels,attr_group_dict=attr_group_dict,testing=True,n_groups=n_groups)
    elif dataset == 'flair':
        # TODO:
        # central: 
        # data file has key: {'metadata', 'train', 'val', 'test'}
            # metadata: {label_counter, fine_grained_label_counter}
            # Note: use np.array() to read in 
        # train: keys() contain all image IDs
        data_dir = os.path.join(data_root, 'flair')
        img_root = os.path.join(data_dir, 'data/small_images')
        label_mapping = None
        fg_label_mapping = None
        
        if args.flair_fine:
            with open(data_dir + '/fine_grained_label_mapping.json') as fg:
                fg_label_mapping = json.load(fg)
        else:
            with open(data_dir + '/label_mapping.json') as f:
                label_mapping = json.load(f)

        trainTransform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize((scale_size, scale_size)),
                                        # transforms.Resize((crop_size, crop_size)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normTransform])

        testTransform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize((scale_size, scale_size)),
                                        transforms.CenterCrop(crop_size),
                                        transforms.ToTensor(),
                                        normTransform])
     
        train_dataset = FlairDataset(split='train', 
                                     num_labels=args.num_labels,
                                     data_file=data_dir,
                                     img_root=img_root,
                                     transform=trainTransform,
                                     label_mapping=label_mapping,
                                     fine_grained_label_mapping=fg_label_mapping,
                                     known_labels=args.train_known_labels)
                                     # modify this, maybe should re-run? (2023.1.13)
        valid_dataset = FlairDataset(split='val', 
                                     num_labels=args.num_labels,
                                     data_file=data_dir,
                                     img_root=img_root,
                                     transform=testTransform,
                                     label_mapping=label_mapping,
                                     fine_grained_label_mapping=fg_label_mapping)
        test_dataset = FlairDataset(split='test', 
                                     num_labels=args.num_labels,
                                     data_file=data_dir,
                                     img_root=img_root,
                                     transform=testTransform,
                                     label_mapping=label_mapping,
                                     fine_grained_label_mapping=fg_label_mapping)
    elif dataset == 'flair_fed':
        # TODO:
        # 1. sample user id (e.g., 200 users per round)
        # 2. for each user, build a model
        # ref: NIID-bench
            # build "net_dataidx_map" for each user (i.e. for each user, it has a dataidx list)
            # get_dataloader returns "train/test_dl_local"
        # Here, we build the dataset to allow the "dataidx"!
        data_dir = os.path.join(data_root, 'flair')
        img_root = os.path.join(data_dir, 'data/small_images')

        label_mapping = None
        fg_label_mapping = None
        
        if args.flair_fine:
            with open(data_dir + '/fine_grained_label_mapping.json') as fg:
                fg_label_mapping = json.load(fg)
        else:
            with open(data_dir + '/label_mapping.json') as f:
                label_mapping = json.load(f)
        trainTransform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize((scale_size, scale_size)),
                                        # transforms.Resize((crop_size, crop_size)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normTransform])

        testTransform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize((scale_size, scale_size)),
                                        transforms.CenterCrop(crop_size),
                                        transforms.ToTensor(),
                                        normTransform])
        
        
        inp_data = h5py.File('/media/liujack/flair_hdf5/fl_data.hdf5', 'r')
        train_dataset = None


        if curr_user != None:
            train_dataset = FlairFedDataset(inp_data=inp_data,
                                        split='train', 
                                        num_labels=args.num_labels,
                                        data_file=data_dir,
                                        img_root=img_root,
                                        curr_user=curr_user,
                                        transform=trainTransform,
                                        label_mapping=label_mapping,
                                        fine_grained_label_mapping=fg_label_mapping,
                                        known_labels=args.train_known_labels)
        else:
            train_dataset = inp_data
        # client agnoistic dataset
        valid_dataset = FlairDataset(split='val', 
                                     num_labels=args.num_labels,
                                     data_file=data_dir,
                                     img_root=img_root,
                                     transform=testTransform,
                                     label_mapping=label_mapping,
                                     fine_grained_label_mapping=fg_label_mapping)
        # client agnoistic dataset
        test_dataset = FlairDataset(split='test', 
                                     num_labels=args.num_labels,
                                     data_file=data_dir,
                                     img_root=img_root,
                                     transform=testTransform,
                                     label_mapping=label_mapping,
                                     fine_grained_label_mapping=fg_label_mapping)

    else:
        print('no dataset avail')
        exit(0)

    if train_dataset is not None:
        train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True, num_workers=workers,drop_last=drop_last)
    if valid_dataset is not None:
        valid_loader = DataLoader(valid_dataset, batch_size=args.test_batch_size,shuffle=False, num_workers=workers)
    if test_dataset is not None:
        test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size,shuffle=False, num_workers=workers)
    if dataset in ['flair_fed']:
        return train_loader, valid_loader, test_loader, train_dataset
    return train_loader,valid_loader,test_loader
