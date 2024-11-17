


import torch
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
from glob import glob
from PIL import Image
from monai.transforms import Compose, LoadImaged, AddChanneld,  RandRotated, RandFlipd, RandZoomd,NormalizeIntensityd,EnsureTyped,RandGaussianSmoothd,EnsureChannelFirstd,Resized
from monai.data import CacheDataset, list_data_collate
import pdb
import pickle
import random
from test import merge_patches
import cv2
def load_pickle(filename="split_dataset.pickle"):
    """
    :param filename: pickle name
    :return: dictionary or list
    """
    with open(filename, "rb") as handle:
        ids = pickle.load(handle)
    return ids

class SAR_Water_Dataset(Dataset):
    def __init__(self, stage,data_mode,mode,data_split,extra_input = None,split_patch_size = None,patch_size = None,stride = None):
        """
        @description  :
        ---------
        @param  :data_mode:train or val or test dataset
        -------
        @param  :mode:train or val or test mode
        @Returns  :
        -------
        """
        
        super(SAR_Water_Dataset, self).__init__()
        
        self.image_paths = load_pickle(data_split)[data_mode][:1]
        self.split_patch_size = split_patch_size
        self.patch_size = patch_size
        self.stride = stride
        self.stage = stage
        self.extra_input = extra_input
        # self.wr_image_dict = {}
        # self.wr_mask_dict = {}
        self.w_image_dict = {}
        self.w_mask_dict = {}
        self.w_pred_dict = {}
        self.mode=mode
        self.data_mode = data_mode
        self.resize =  Resized(keys=["image", "label"],spatial_size = self.patch_size,mode = ["area","nearest"])
        self.train_trasnform = Compose(
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
        self.test_transform = Compose(
        [
            # EnsureChannelFirstd(keys=["image", "label"]),
            # NormalizeIntensityd(keys=["image"]),
            EnsureTyped(keys=["image", "label"]),
        ]
    )
        # 1.-load dataset/save whole-size images and masks
        for i in self.image_paths:
            imgname = i.split("/")[-1].split(".")[0]
            mask_path = os.path.join("Dataset/PCA_SWM/GT",imgname+"binary.tif")
            image, mask = self.load_data(i, mask_path)
            mask = 255-mask
            if self.stage =="c":
                patches_image, patches_mask,_ = self.get_patches(imgname,image, mask,extra_input = self.extra_input,patch_size = self.split_patch_size)
                self.w_image_dict.update(patches_image)
                self.w_mask_dict.update(patches_mask)
            else:
                patches_image, patches_mask,patches_pred = self.get_patches(imgname,image, mask,extra_input = self.extra_input[imgname],patch_size = self.patch_size)
                self.w_image_dict.update(patches_image)
                self.w_mask_dict.update(patches_mask)
                self.w_pred_dict.update(patches_pred)
            
            print(f"stage:{stage},mode:{mode}, casename:{i},split_num:{len(patches_image)}")
            # self.w_image_dict[imgname]= image
            # self.w_mask_dict[imgname]= mask
            
            
            # patches_image, patches_mask = self.get_patches(imgname,image, mask)
            # self.image_dict.update(patches_image)
            # self.mask_dict.update(patches_mask)
            # print(f"{mode}, casename:{i},split_num:{len(patches_image)}")
        
        if mode =="train":
            dd = list(self.w_image_dict.items())
            print(f"total_number:{len(dd)}")
            random.shuffle(dd)
            self.w_image_dict = dict(dd)
    def __len__(self):
        return len(self.w_image_dict)

    def __getitem__(self, index):
        # pdb.set_trace()
        w_image = self.w_image_dict[list(self.w_image_dict.keys())[index]]
        w_mask = self.w_mask_dict[list(self.w_image_dict.keys())[index]]
        
        w_image = w_image/255.0
        w_mask = w_mask/255.0
        w_image = torch.tensor(np.expand_dims(w_image,0))
        w_mask =  torch.tensor(np.expand_dims(w_mask,0))
        
        
        if self.stage =="c":
            zero_input = torch.zeros_like(torch.tensor(w_image))
            w_image = torch.cat([w_image] * 3, dim=0)
            w_image = torch.cat([w_image,zero_input] , dim=0)#channel = 4
            
        else: 
            w_image = torch.cat([w_image] * 3, dim=0)
            pred_patch = self.w_pred_dict[list(self.w_image_dict.keys())[index]]
            w_image = torch.cat([w_image,pred_patch] , dim=0)#channel = 4
        if self.mode=="train":
            train_pair = {"image":w_image,"label":w_mask}
            w_pair = self.train_trasnform (train_pair)
            if self.stage =="c":
                w_pair = self.resize(w_pair)
        else:
            test_pair = {"image":w_image,"label":w_mask}
            w_pair = self.test_transform (test_pair)
            if self.stage =="c":
                w_pair = self.resize(w_pair)
        return list(self.w_image_dict.keys())[index],w_pair["image"].float(),w_pair["label"].float()



    def get_patches(self, imgname,image, mask,extra_input = None,patch_size = None):
        # print(image.shape)
        # Split image and mask into patches using sliding window approach
        height, width = image.shape
        patch_height, patch_width = patch_size
        stride_height, stride_width = self.stride

        # Calculate the number of patches in each dimension

        num_patches_height = (height - patch_height) // stride_height + 2
        num_patches_width = (width - patch_width) // stride_width + 2

        # Initialize the patches array with zeros
        patches_image = {}
        patches_mask = {}
        patches_pred = {}
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
                if extra_input!=None:
                    patch_pred = extra_input[start_x:end_x, start_y:end_y]
                    patches_pred[f"{imgname}_{start_x}_{start_y}"]= patch_pred

        return patches_image, patches_mask,patches_pred
    

    def load_data(self, image_path, mask_path):
        # Load image and mask data using PIL
        image_data = np.array(Image.open(image_path))
        mask_data = np.array(Image.open(mask_path))
        # print(image_path,mask_path)
        # print(image_data.shape,mask_data.shape)
        assert image_data.shape==mask_data.shape
        
        return image_data, mask_data


    
if __name__ =="__main__":

    image_path = "Dataset/PCA_SWM/waterdata"
    mask_path = "Dataset/PCA_SWM/GT"
    image_paths = sorted(glob(os.path.join(image_path,"*.jpg"))[:1])
    mask_paths = sorted(glob(os.path.join(mask_path,"*binary.tif"))[:1])
    # image_data = np.array(Image.open(mask_paths[0]))
    # print(image_data.shape)
    # print(len(image_paths),len(mask_paths))
    SARtrain_dataset = SAR_Water_Dataset(stage = "c",data_mode = "test",mode = "test",data_split= "SAR_water_1.pickle",split_patch_size= (512,512),patch_size=(128, 128), stride=(128,128))
    dataloader = DataLoader(SARtrain_dataset, batch_size=4, shuffle=True, num_workers=0, collate_fn=list_data_collate)


    from models.wingnet import Wingnet_encoder,Wingnet_decoder
    from models.CToF import SegmentationModel
    from torchvision import transforms
    from torch.nn import DataParallel
    encoder = Wingnet_encoder(in_channel=4,n_classes=1)
    c_decoder = Wingnet_decoder(n_classes=1)
    f_decoder = Wingnet_decoder(n_classes=1)
    coarse_model = SegmentationModel(encoder=encoder,decoder=c_decoder)
   
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # coarse_model = DataParallel(coarse_model)
    coarse_model.to(device)
    coarse_model.eval()
    # pdb.set_trace()
    all_preds = {}
    all_labels = {}
    for i,(name,w_image, w_mask) in enumerate(dataloader):
        
        inputs = w_image.to(device)
        labels = w_mask.to(device)
        outputs = labels
        outputs = coarse_model(inputs)
        print(inputs.shape,labels.shape,inputs.device)
        resize = transforms.Resize([w_mask.shape[-2],w_mask.shape[-1]])
        w_pred = resize(outputs[0])
        for j in range(inputs.shape[0]):
            y_pred = w_pred[j, 0]
            all_preds[name[j]] = y_pred
            all_labels[name[j]] = labels[j, 0]
        
        # print(name)
        # w_pred = w_mask
    merged_preds = merge_patches(device,all_preds,patch_shape= (inputs.shape[-2],inputs.shape[-1]),img_shape = (6944,6438))
    # merged_labels = merge_patches(device,all_labels,patch_shape= (inputs.shape[-2],inputs.shape[-1]),img_shape = (6944,6438))
    patchSARtrain_dataset = SAR_Water_Dataset(stage = "f",data_mode = "test",mode = "test",extra_input = merged_preds,data_split= "SAR_water_1.pickle",split_patch_size= (512,512),patch_size=(128, 128), stride=(128,128))
    patchdataloader = DataLoader(patchSARtrain_dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=list_data_collate)
    for j,(patch_name,patch_image,patch_mask) in enumerate(patchdataloader):
        # print(name)
        print(patch_image.shape)
        print(patch_mask.shape)
        