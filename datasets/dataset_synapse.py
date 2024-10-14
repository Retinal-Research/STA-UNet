import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from batchgenerators.augmentations.spatial_transformations import augment_spatial_2
# from torchvision.transforms import v2
from torchvision.transforms import functional as VF
from PIL import Image
import warnings 

# def rgb_to_class_mask(image_path, dataset, size):
#     # Define the class mapping
#     if dataset == 'ASSD':
#         class_mapping = {
#             'paved-area': (128, 64, 128),
#             'dirt': (130, 76, 0),
#             'grass': (0, 102, 0),
#             'gravel': (112, 103, 87),
#             'water': (28, 42, 168),
#             'rocks': (48, 41, 30),
#             'pool': (0, 50, 89),
#             'vegetation': (107, 142, 35),
#             'roof': (70, 70, 70),
#             'wall': (102, 102, 156),
#             'window': (254, 228, 12),
#             'door': (254, 148, 12),
#             'fence': (190, 153, 153),
#             'fence-pole': (153, 153, 153),
#             'person': (255, 22, 96),
#             'dog': (102, 51, 0),
#             'car': (9, 143, 150),
#             'bicycle': (119, 11, 32),
#             'tree': (51, 51, 0),
#             'bald-tree': (190, 250, 190),
#             'ar-marker': (112, 150, 146),
#             'obstacle': (2, 135, 115),
#             'conflicting': (255, 0, 0),
#         }
    
#     # Create a reverse mapping for quick lookup
#     reverse_mapping = {v: k for k, v in class_mapping.items()}
#     class_to_index = {k: i for i, k in enumerate(class_mapping.keys())}

#     # Open the image
#     img = Image.open(image_path).convert('RGB')
#     img = img.resize(size, Image.NEAREST)  # Use nearest neighbor for masks to avoid interpolation issues
#     img_array = np.array(img)
    
#     # Initialize a single-channel mask
#     class_mask = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=np.float32)
    
#     # Map each pixel to the class index
#     for rgb, class_idx in class_to_index.items():
#         mask = np.all(img_array == np.array(rgb), axis=-1)
#         class_mask[mask] = float(class_idx)
    
#     return class_mask

import numpy as np
import warnings

# Suppress FutureWarnings in a specific block
def rgb_to_class_mask(image_path, dataset, size):
    # Define the class mapping
    if dataset == 'ASSD':
        class_mapping = {
            'paved-area': (128, 64, 128),
            'dirt': (130, 76, 0),
            'grass': (0, 102, 0),
            'gravel': (112, 103, 87),
            'water': (28, 42, 168),
            'rocks': (48, 41, 30),
            'pool': (0, 50, 89),
            'vegetation': (107, 142, 35),
            'roof': (70, 70, 70),
            # 'wall': (102, 102, 156),
            # 'window': (254, 228, 12),
            # 'door': (254, 148, 12),
            # 'fence': (190, 153, 153),
            # 'fence-pole': (153, 153, 153),
            # 'person': (255, 22, 96),
            # 'dog': (102, 51, 0),
            # 'car': (9, 143, 150),
            # 'bicycle': (119, 11, 32),
            # 'tree': (51, 51, 0),
            # 'bald-tree': (190, 250, 190),
            # 'ar-marker': (112, 150, 146),
            # 'obstacle': (2, 135, 115),
            # 'conflicting': (255, 0, 0),
        }
    
    # Open the image
    img = Image.open(image_path).convert('RGB')
    img = img.resize(size, Image.NEAREST)  # Use nearest neighbor for masks to avoid interpolation issues
    img_array = np.array(img)

    # Initialize masks
    num_classes = len(class_mapping)
    masks = np.zeros((num_classes, img_array.shape[0], img_array.shape[1]), dtype=np.uint8)

    # Create a reverse mapping for quicker lookup
    reverse_mapping = {v: k for k, v in class_mapping.items()}

    # Suppress FutureWarnings in this block
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        # Iterate through each pixel and create masks
        for class_idx, (class_name, rgb) in enumerate(class_mapping.items()):
            if class_name == 'unlabeled':
                continue
            masks[class_idx] = np.all(img_array == np.array(rgb), axis=-1).astype(np.uint8)
    
    return masks

def dino_augmentation(image,label):
    #print(label.shape)
    #print(image.shape)
    patch_shape = label.shape


    image = image.reshape(1,1,512,512)
    label = label.reshape(1,1,512,512)

    image, label = augment_spatial_2(image,label,patch_size=patch_shape,do_elastic_deform=True,do_rotation=False,do_scale=False,random_crop=False)

    image = image.reshape(512,512)
    label = label.reshape(512,512)

    return image,label

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

def get_random_elastic_params(alpha, sigma, size):
    # _, h, w = VF.get_dimensions(image)
    # size = [h, w]
    alpha = 20.0
    sigma = 5.0
    dx = torch.rand([1, 1] + size) * 2 -1
    if sigma > 0:
        kx = int(8 * sigma + 1)
        if kx % 2 == 0:
            kx += 1
        dx = VF.gaussian_blur(dx, [kx, kx], sigma)
    dx = dx * alpha / size[0]

    dy = torch.rand([1, 1] + size) * 2 -1
    if sigma > 0:
        ky = int(8 * sigma + 1)
        if ky % 2 == 0:
            ky += 1
        dy = VF.gaussian_blur(dy, [ky, ky], sigma)
    dy = dy * alpha / size[1]
    return torch.concat([dx, dy], 1).permute([0, 2, 3, 1])

# def elastic


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)

        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample

class RandomGenerator_DINO(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        #elif random.random() > 0.5:
            #image,label = dino_augmentation(image,label)

        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))

        # if random.random() > 0.5:
        #     image = v2.ElasticTransform()(image)

        sample = {'image': image, 'label': label.long()}
        return sample


class RandomGenerator_DINO_Deform(object):
    def __init__(self, output_size,alpha=20.,sigma=5.):
        self.output_size = output_size
        self.alpha = 20.0
        self.sigma = 5.0

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        #elif random.random() > 0.5:
            #image,label = dino_augmentation(image,label)

        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0) # Bx1x224x224
        label = torch.from_numpy(label.astype(np.float32)) # Bx224x224

        # if random.random() > 0.5:
        _, h, w = VF.get_dimensions(image)
        size = [int(h), int(w)]
        # print(size)
        displacement = get_random_elastic_params(self.alpha, self.sigma, size)
        # print(displacement.shape)
        image_dino = VF.elastic_transform(image, displacement, VF.InterpolationMode.BILINEAR, 0)
        # label_dino = VF.elastic_transform(label, displacement, VF.InterpolationMode.NEAREST, 0)
        # image 
        sample = {'image': image, 'label': label.long(), 'image_dino': image_dino, 'disp':displacement}
        return sample

class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None,transform_dino=None):
        self.transform = transform  # using transform in torch!
        self.transform_dino = transform_dino
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
            # dino_image,dino_label = data['image'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]
            # dino_image,dino_label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}
        # dino_sample = {'image': dino_image, 'label':dino_label}

        if self.transform:
            sample = self.transform(sample)

            # dino_sample = self.transform_dino(dino_sample)
            # dino_sample = self.transform_dino(dino_sample)

        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample#,dino_sample

class ASSD_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None,transform_dino=None):
        self.transform = transform  # using transform in torch!
        self.transform_dino = transform_dino
        self.split = split
        self.image_list = os.listdir(base_dir + '/' + 'train_images')
        self.data_dir = base_dir

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        if self.split == "train":
            image_name = self.image_list[idx]
            image_path = self.data_dir + '/' + 'train_images/' + image_name
            image = Image.open(image_path).convert('L')
            image = np.array(image)
            label_path = self.data_dir + '/' + 'color_image_masks_train/' + image_name[:-3] + 'png'
            label = rgb_to_class_mask(label_path, dataset= 'ASSD', size= [512,512])
            # image, label = data['image'], data['label']
            # dino_image,dino_label = data['image'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]
            # dino_image,dino_label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}
        # dino_sample = {'image': dino_image, 'label':dino_label}

        # if self.transform:
        #     sample = self.transform(sample)

            # dino_sample = self.transform_dino(dino_sample)
            # dino_sample = self.transform_dino(dino_sample)

        sample['case_name'] = image_name
        return sample#,dino_sample