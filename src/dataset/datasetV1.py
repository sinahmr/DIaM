import argparse
import random
from multiprocessing import Pool
from typing import List
import sys
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import src.dataset.transform as transform
from .classes import get_split_classes
from .utils import make_dataset
from torchvision.transforms.functional import resize
# from torchvision import transforms

def get_val_loader(cfg: dict, args: argparse.Namespace) -> torch.utils.data.DataLoader:
    """
        Build the validation loader.
    """
    assert args.split in [0, 1, 2, 3, 10, 11, -1]
    val_transform = transform.Compose([
            transform.Resize(cfg['DATA']['image_size']),
            transform.ToTensor(),
            transform.Normalize(mean=cfg['DATA']['mean'], std=cfg['DATA']['std'])])

    split_classes = get_split_classes(cfg, args)
    # ===================== Get base and novel classes =====================
    # print(f"Data: {cfg['DATA']['data_name']}, S{cfg['DATA']['split']}") 
    # print(f'Data: {cfg['DATA']['data_name']}, S{cfg['DATA']['split']}')
    
    base_class_list = split_classes[cfg['DATA']['data_name']][cfg['DATA']['split']]['train']
    novel_class_list = split_classes[cfg['DATA']['data_name']][cfg['DATA']['split']]['val']
    print('Novel classes:', novel_class_list)
    print('Base classes:', base_class_list)
    args.num_classes_tr = len(base_class_list) + 1  # +1 for bg
    args.num_classes_val = len(novel_class_list)
    print(f"Novel classes {args.num_classes_val}  {args.num_classes_tr}") # Add this line
    # sys.exit(1)
    # ===================== Build loader =====================
    val_sampler = None
    val_data = MultiClassValData(transform=val_transform,
                                 base_class_list=base_class_list,
                                 novel_class_list=novel_class_list,
                                 data_list_path_train=cfg['DATA']['train_list'],
                                 data_list_path_test=cfg['DATA']['val_list'],
                                 args=args,
                                 cfg=cfg)
                          
    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=cfg['EVALUATION']['batch_size_val'],
                                             drop_last=False,
                                             shuffle=cfg['EVALUATION']['shuffle_test_data'],
                                             num_workers=cfg['DATA']['workers'],
                                             pin_memory=cfg['DATA']['pin_memory'],
                                             sampler=val_sampler)
  
    # prepare data iterator

    print(f'number of novel class in dataset preparation... {len(val_data.novel_class_list)}')
    print(f'number of ALL class in dataset preparation... {len(val_data.all_classes)}')
    args.num_novel_classes = len(val_data.novel_class_list)
    total_samples = len(val_loader)
    # print(f"Total data samples: {total_samples}")
    return val_loader
# data/coco/val2014/
def get_image_and_label(image_path, label_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.float32(image)
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
        raise (RuntimeError("Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
    return image, label

def adjust_label(base_class_list, novel_class_list, label, chosen_novel_class,  base_label=-1, other_novels_label=255):
    # -1 for base_label or other_novels_label means including the true labels
    assert base_label in [-1, 0, 255] and other_novels_label in [-1, 0, 255]
    new_label = np.zeros_like(label)  # background
    for lab in base_class_list:
        indexes = np.where(label == lab)
        if base_label == -1:
            new_label[indexes[0], indexes[1]] = base_class_list.index(lab) + 1  # Add 1 because class 0 is bg
        else:
            new_label[indexes[0], indexes[1]] = base_label

    for lab in novel_class_list:
        indexes = np.where(label == lab)
        if other_novels_label == -1:
            new_label[indexes[0], indexes[1]] = 1 + len(base_class_list) + novel_class_list.index(lab)
        elif lab == chosen_novel_class:
            new_label[indexes[0], indexes[1]] = 1 + len(base_class_list)
        else:
            new_label[indexes[0], indexes[1]] = other_novels_label

    ignore_pix = np.where(label == 255)
    new_label[ignore_pix] = 255

    return new_label


class ClassicValData(Dataset):
    def __init__(self, transform: transform.Compose, base_class_list: List[int], novel_class_list: List[int],
                 data_list_path_train: str, data_list_path_test: str, args: argparse.Namespace):
        assert args.support_only_one_novel
        self.shot = args.shot
        self.data_root = args.data_root
        self.base_class_list = base_class_list
        self.novel_class_list = novel_class_list
        self.transform = transform

        self.use_training_images_for_supports = args.use_training_images_for_supports
        assert not self.use_training_images_for_supports or data_list_path_train
        support_data_list_path = data_list_path_train if self.use_training_images_for_supports else data_list_path_test

        self.query_data_list, _ = make_dataset(args.data_root, data_list_path_test,
                                               self.base_class_list + self.novel_class_list,
                                               keep_small_area_classes=True)
        print('Total number of kept images (query):', len(self.query_data_list))
        self.support_data_list, self.support_sub_class_file_list = make_dataset(args.data_root, support_data_list_path,
                                                                                self.novel_class_list,
                                                                                keep_small_area_classes=False)
        print('Total number of kept images (support):', len(self.support_data_list))

    @property
    def num_novel_classes(self):
        return len(self.novel_class_list)

    @property
    def all_classes(self):
        return [0] + self.base_class_list + self.novel_class_list

    def _adjust_label(self, label, chosen_novel_class,  base_label=-1, other_novels_label=255):
        return adjust_label(self.base_class_list, self.novel_class_list,
                            label, chosen_novel_class,  base_label, other_novels_label)

    def __len__(self):
        return len(self.query_data_list)

    def __getitem__(self, index):
        # ========= Read query image and Choose class =======================
        image_path, label_path = self.query_data_list[index]
        qry_img, label = get_image_and_label(image_path, label_path)
        
        # # ===========  RESIZE QUERY IMAGE HERE =============
        # desired_output_size = (26, 26)  # Example: Set your desired output size
        # required_input_size = (desired_output_size[0] * 8, 
        #                        desired_output_size[1] * 8)
        # qry_img = cv2.resize(qry_img, required_input_size, interpolation=cv2.INTER_LINEAR) 
        if self.transform is not None:
            qry_img, label = self.transform(qry_img, label)
            # print(f"Query image and label: {qry_img.shape}.....{label.shape}") # Add this line
        # desired_output_size = (26, 26)  # Example: Set your desired output size
        # required_input_size = (desired_output_size[0] * 8, 
        #                        desired_output_size[1] * 8)
        
        # # Resize using PyTorch
        # qry_img = resize(qry_img, required_input_size, interpolation=transforms.InterpolationMode.BILINEAR) 
        # print(f"Query image after transform shape: {qry_img.shape}") # Add this line
        # == From classes in the query image, choose one randomly ===
        label_class = set(np.unique(label))
        label_class -= {0, 255}
        novel_classes_in_image = list(label_class.intersection(set(self.novel_class_list)))
        if len(novel_classes_in_image) > 0:
            class_chosen = np.random.choice(novel_classes_in_image)
        else:
            class_chosen = np.random.choice(self.novel_class_list)

        q_valid_pixels = (label != 255).float()
        target = self._adjust_label(label, class_chosen, base_label=-1, other_novels_label=0)

        support_image_list = []
        support_label_list = []

        file_class_chosen = self.support_sub_class_file_list[class_chosen]
        num_file = len(file_class_chosen)

        # ========= Build support ==============================================
        # == First, randomly choose indexes of support images ==
        support_image_path_list = []
        support_label_path_list = []
        support_idx_list = []

        for _ in range(self.shot):
            support_idx = random.randint(1, num_file) - 1
            support_image_path = image_path
            support_label_path = label_path
            while (support_image_path == image_path and support_label_path == label_path) or support_idx in support_idx_list:
                support_idx = random.randint(1, num_file) - 1
                support_image_path, support_label_path = file_class_chosen[support_idx]
            support_idx_list.append(support_idx)
            support_image_path_list.append(support_image_path)
            support_label_path_list.append(support_label_path)

        # == Second, read support images and masks  ============
        for k in range(self.shot):
            support_image_path, support_label_path = support_image_path_list[k], support_label_path_list[k]
            support_image, support_label = get_image_and_label(support_image_path, support_label_path)
            support_label = self._adjust_label(support_label, class_chosen, base_label=0, other_novels_label=0)
            support_image_list.append(support_image)
            support_label_list.append(support_label)

        # == Forward images through transforms =================
        if self.transform is not None:
            for k in range(len(support_image_list)):
                support_image_list[k], support_label_list[k] = self.transform(support_image_list[k], support_label_list[k])
                support_image_list[k] = support_image_list[k].unsqueeze(0)
                support_label_list[k] = support_label_list[k].unsqueeze(0)

        # == Reshape properly ==================================
        spprt_imgs = torch.cat(support_image_list, 0)
        spprt_labels = torch.cat(support_label_list, 0)

        return qry_img, target, q_valid_pixels, spprt_imgs, spprt_labels, class_chosen

class MultiClassValData(Dataset):
    def __init__(self, transform: transform.Compose, base_class_list: List[int], novel_class_list: List[int],
                 data_list_path_train: str, data_list_path_test: str, args: argparse.Namespace, cfg: dict):
        self.support_only_one_novel = cfg['EVALUATION']['support_only_one_novel']
        self.use_training_images_for_supports = cfg['EVALUATION']['use_training_images_for_supports']
        assert not self.use_training_images_for_supports or data_list_path_train
        support_data_list_path = data_list_path_train if self.use_training_images_for_supports else data_list_path_test

        self.shot = cfg['EVALUATION']['shot']
        self.data_root = cfg['DATA']['data_root']
        self.base_class_list = base_class_list  # Does not contain bg
        self.novel_class_list = novel_class_list
        self.query_data_list, _ = make_dataset(cfg['DATA']['data_root'], data_list_path_test,
                                               self.base_class_list + self.novel_class_list,
                                               keep_small_area_classes=True)
        self.complete_query_data_list = self.query_data_list.copy()
        print('Total number of kept images (query)-MULTICLASS:', len(self.query_data_list))
        support_data_list, self.support_sub_class_file_list = make_dataset(cfg['DATA']['data_root'], support_data_list_path,
                                                                           self.novel_class_list,
                                                                           keep_small_area_classes=False)
        print('Total number of kept images (support):', len(support_data_list))
        self.transform = transform

    @property
    def num_novel_classes(self):
        return len(self.novel_class_list)

    @property
    def all_classes(self):
        return [0] + self.base_class_list + self.novel_class_list

    def _adjust_label(self, label, chosen_novel_class, base_label=-1, other_novels_label=255):
        return adjust_label(self.base_class_list, self.novel_class_list,
                            label, chosen_novel_class, base_label, other_novels_label)

    def __len__(self):
        return len(self.query_data_list)

    def __getitem__(self, index):  # It only gives the query
        image_path, label_path = self.query_data_list[index]
        qry_img, label = get_image_and_label(image_path, label_path)
        # print(f"Query image original shape: {qry_img.shape}") # Add this line
        
        label = self._adjust_label(label, -1, base_label=-1, other_novels_label=-1)
        if self.transform is not None:
            qry_img, label = self.transform(qry_img, label)
           
        # print(f"Query image after transform shape: {qry_img.shape}") # Add this line torch.Size([3, 417, 417])
        valid_pixels = (label != 255).float()
        # query image get item shape ..torch.Size([3, 417, 417]) and the ..... lable ... torch.Size([417, 417])
        # print(f'query image get item shape ..{qry_img.shape} and the ..... lable ... {label.shape}')
        return qry_img, label, valid_pixels, image_path

    def generate_support(self, query_image_path_list, remove_them_from_query_data_list=False):
        # print("GENERATION_SUPPORT_IMAGES......")
        image_list, label_list = list(), list()
        support_image_path_list, support_label_path_list = list(), list()
        # print(f"Number of novel classes before: {len(self.novel_class_list)}")
        for c in self.novel_class_list:
            file_class_chosen = self.support_sub_class_file_list[c]
            num_file = len(file_class_chosen)
            indices_list = list(range(num_file))
            random.shuffle(indices_list)
            current_path_list = list()
            for idx in indices_list:
                if len(current_path_list) >= self.shot:
                    break
                image_path, label_path = file_class_chosen[idx]
                if image_path in (query_image_path_list + current_path_list):
                    continue
                image, label = get_image_and_label(image_path, label_path)
                # print(f'image after resizing is...{image.shape}')
                # print(f'Support image original shape: {image.shape}')  # Add this line

                if self.support_only_one_novel:  # Ignore images that have multiple novel classes
                    present_novel_classes = set(np.unique(label)) - {0, 255} - set(self.base_class_list)
                    if len(present_novel_classes) > 1:
                        continue
               
                label = self._adjust_label(label, -1, base_label=0, other_novels_label=-1)  # If support_only_one_novel is True, images with more than one novel classes won't reach this line. So, -1 won't make the image contain two different novel classes.
                image_list.append(image)
                label_list.append(label)
                current_path_list.append(image_path)
                support_image_path_list.append(image_path)
                support_label_path_list.append(label_path)
            found_images_count = len(current_path_list)
            assert found_images_count > 0, f'No support candidate for class {c} out of {num_file} images'
            if found_images_count < self.shot:
                indices_to_repeat = random.choices(range(found_images_count), k=self.shot-found_images_count)
                image_list.extend([image_list[i] for i in indices_to_repeat])
                label_list.extend([label_list[i] for i in indices_to_repeat])

        transformed_image_list, transformed_label_list = list(), list()
        if self.shot == 1:
            for i, l in zip(image_list, label_list):
                transformed_i, transformed_l = self.transform(i, l)
                # print(f"Support image after transform: {transformed_i.shape}") 
                transformed_image_list.append(transformed_i.unsqueeze(0))
                transformed_label_list.append(transformed_l.unsqueeze(0))
        else:
            with Pool(self.shot) as pool:
                for transformed_i, transformed_l in pool.starmap(self.transform, zip(image_list, label_list)):
                    transformed_image_list.append(transformed_i.unsqueeze(0))
                    transformed_label_list.append(transformed_l.unsqueeze(0))
                pool.close()
                pool.join()

        spprt_imgs = torch.cat(transformed_image_list, 0)
        spprt_labels = torch.cat(transformed_label_list, 0)

        if remove_them_from_query_data_list and not self.use_training_images_for_supports:
            self.query_data_list = self.complete_query_data_list.copy()
            for i, l in zip(support_image_path_list, support_label_path_list):
                self.query_data_list.remove((i, l))

        print("RETURNING GENERATE IMAGES.....")
        print(f'{spprt_labels.shape}')
        # Query image after transform shape generation support section: torch.Size([20, 3, 417, 417]) and label .....torch.Size([20, 417, 417])
        print(f"Query image after transform shape generation support section: {spprt_imgs.shape} and label .....{spprt_labels.shape}")
        return spprt_imgs, spprt_labels
