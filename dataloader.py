


import torch
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
from glob import glob
from PIL import Image
from monai.transforms import Compose, RandRotated, RandFlipd, RandZoomd,NormalizeIntensityd,EnsureTyped,RandGaussianSmoothd
from monai.data import CacheDataset, list_data_collate
import pdb
import pickle
import random
def load_pickle(filename="split_dataset.pickle"):
    """
    :param filename: pickle name
    :return: dictionary or list
    """
    with open(filename, "rb") as handle:
        ids = pickle.load(handle)
    return ids

class SAR_Water_Dataset(Dataset):
    def __init__(self, mode,data_split, patch_size=(64, 64), stride=(32, 32)):
        super(SAR_Water_Dataset, self).__init__()
        
        self.image_paths = load_pickle(data_split)[mode]
        self.patch_size = patch_size
        self.stride = stride
        self.image_dict = {}
        self.mask_dict = {}
        self.mode=mode
        
        for i in self.image_paths:
            imgname = i.split("/")[-1].split(".")[0]
            mask_path = os.path.join("Dataset/PCA_SWM/GT",imgname+"binary.tif")
            image, mask = self.load_data(i, mask_path)
            patches_image, patches_mask = self.get_patches(imgname,image, mask)
            self.image_dict.update(patches_image)
            self.mask_dict.update(patches_mask)
            print(f"{mode}, casename:{i},split_num:{len(patches_image)}")
        
        if mode =="train":
            dd = list(self.image_dict.items())
            print(f"total_number:{len(dd)}")
            random.shuffle(dd)
            self.image_dict = dict(dd)
    def __len__(self):
        return len(self.image_dict)

    def __getitem__(self, index):
        # pdb.set_trace()
        image_patch = self.image_dict[list(self.image_dict.keys())[index]]
        mask_patch = self.mask_dict[list(self.image_dict.keys())[index]]
        image_patch = image_patch/255.0
        mask_patch = mask_patch/255.0
        image_patch = np.expand_dims(image_patch,0)
        mask_patch = np.expand_dims(mask_patch,0)
        if self.mode=="train":

            train_pair = {"image":image_patch,"label":mask_patch}
            train_transforms = Compose(
        [
            # EnsureChannelFirstd(keys=["image", "label"]),
            # AddChanneld(keys=["image", "label"]),
            RandFlipd(keys=["image", "label"],prob=0.5, spatial_axis=0),
            RandRotated(keys=["image", "label"],range_x=10, range_y=10, range_z=10, prob=0.5),
            RandZoomd(keys=["image", "label"],min_zoom=0.9, max_zoom=1.1, prob=0.5, keep_size=True),
            RandGaussianSmoothd(
                keys=["image"],
                # sigma = (0.25,1.5),
                sigma_x=(0.25, 1.5),
                sigma_y=(0.25, 1.5),
                sigma_z=(0.25, 1.5),
                approx="erf",
                prob=0.1,
                allow_missing_keys=False,
            ),
            # NormalizeIntensityd(keys=["image"]),
            EnsureTyped(keys=["image", "label"]),
        ]
    )
            # Apply transformations to image and mask patches
            patch_pair = train_transforms(train_pair)
            image_patch, mask_patch = patch_pair["image"],patch_pair["label"]
            image_patch = torch.cat([image_patch] * 3, dim=0)
            
            return list(self.image_dict.keys())[index],image_patch.float(), mask_patch.float()
        else:
            test_pair = {"image":image_patch,"label":mask_patch}
            test_transforms = Compose(
        [
            # EnsureChannelFirstd(keys=["image", "label"]),
            # NormalizeIntensityd(keys=["image"]),
            EnsureTyped(keys=["image", "label"]),
        ]
    )
            # Apply transformations to image and mask patches
            patch_pair = test_transforms(test_pair)
            image_patch, mask_patch = patch_pair["image"],patch_pair["label"]
            image_patch = torch.cat([image_patch] * 3, dim=0)
            
            return list(self.image_dict.keys())[index],image_patch.float(), mask_patch.float()
           


    def load_data(self, image_path, mask_path):
        # Load image and mask data using PIL
        image_data = np.array(Image.open(image_path))
        mask_data = np.array(Image.open(mask_path))
        # print(image_path,mask_path)
        # print(image_data.shape,mask_data.shape)
        assert image_data.shape==mask_data.shape
        
        return image_data, mask_data

    def get_patches(self, imgname,image, mask):
        # print(image.shape)
        # Split image and mask into patches using sliding window approach
        height, width = image.shape
        patch_height, patch_width = self.patch_size
        stride_height, stride_width = self.stride

        # Calculate the number of patches in each dimension

        num_patches_height = (height - patch_height) // stride_height + 2
        num_patches_width = (width - patch_width) // stride_width + 2

        # Initialize the patches array with zeros
        patches_image = {}
        patches_mask = {}

        # Fill the patches array by iterating over the image with the sliding window
        for i in range(num_patches_height):
            for j in range(num_patches_width):
                start_x = i * stride_height
                end_x = i * stride_height + patch_height
                start_y= j * stride_width
                end_y = j * stride_width + patch_width
                if end_x>height:
                    end_x = height
                    start_x = height-patch_height
                if end_y>width:
                    end_y = width
                    start_y = width-patch_width
                # print(end_x,end_y)
                patch_image = image[start_x:end_x, start_y:end_y]
                patch_mask = mask[start_x:end_x, start_y:end_y]
                #qufan
                patch_mask = 255-patch_mask
                # if self.mode=="train" and patch_mask.sum()==0:
                #     pass
                # else:
                patches_image[f"{imgname}_{start_x}_{start_y}"]= patch_image
                patches_mask[f"{imgname}_{start_x}_{start_y}"]= patch_mask

        return patches_image, patches_mask
    
    

    
if __name__ =="__main__":

    image_path = "Dataset/PCA_SWM/waterdata"
    mask_path = "Dataset/PCA_SWM/GT"
    image_paths = sorted(glob(os.path.join(image_path,"*.jpg"))[:1])
    mask_paths = sorted(glob(os.path.join(mask_path,"*binary.tif"))[:1])
    # image_data = np.array(Image.open(mask_paths[0]))
    # print(image_data.shape)
    # print(len(image_paths),len(mask_paths))
    SARtrain_dataset = SAR_Water_Dataset("train",data_split= "SAR_water_1.pickle",patch_size=(256,256), stride=(128, 128))
    dataloader = DataLoader(SARtrain_dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=list_data_collate)

    # pdb.set_trace()
    for i,(name,img,mask) in enumerate(dataloader):
        print(name)
        print(img.shape)
        print(mask.shape)