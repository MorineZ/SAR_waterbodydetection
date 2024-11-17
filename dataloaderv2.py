


import torch
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
from glob import glob
from PIL import Image
from monai.transforms import Compose, RandRotated, RandFlipd, RandZoomd,NormalizeIntensityd,EnsureTyped,RandGaussianSmoothd,Resized
from monai.data import CacheDataset, list_data_collate
import pdb
import pickle
import random
import cv2
def load_pickle(filename="split_dataset.pickle"):
    """
    :param filename: pickle name
    :return: dictionary or list
    """
    with open(filename, "rb") as handle:
        ids = pickle.load(handle)
    return ids

class Coarse_SAR_Water_Dataset(Dataset):
    def __init__(self, data_mode,mode,data_split,split_patch_size,patch_size,stride):
        """
        @description  :
        ---------
        @param  :data_mode:train or val or test dataset
        -------
        @param  :mode:train or val or test mode
        @Returns  :
        -------
        """
        
        super(Coarse_SAR_Water_Dataset, self).__init__()
        
        self.image_paths = load_pickle(data_split)[data_mode]
        self.split_patch_size = split_patch_size
        self.patch_size = patch_size
        self.stride = stride
        # self.wr_image_dict = {}
        # self.wr_mask_dict = {}
        self.w_image_dict = {}
        self.w_mask_dict = {}
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
            patches_image, patches_mask = self.get_patches(imgname,image, mask)
            self.w_image_dict.update(patches_image)
            self.w_mask_dict.update(patches_mask)
            # print(f"{mode}, casename:{i},split_num:{len(patches_image)}")
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
        w_mask = self.w_mask_dict[list(self.w_mask_dict.keys())[index]]
        w_image = w_image/255.0
        w_mask = w_mask/255.0
        w_image = np.expand_dims(w_image,0)
        w_mask = np.expand_dims(w_mask,0)
        if self.mode=="train":

            train_pair = {"image":w_image,"label":w_mask}
            w_pair = self.train_trasnform(train_pair)
        else:
            test_pair = {"image":w_image,"label":w_mask}
            w_pair = self.test_transform (test_pair)
        w_image, w_mask = w_pair["image"],w_pair["label"]
        wr_pair = self.resize(w_pair)
        wr_image, wr_mask = wr_pair["image"],wr_pair["label"]
        w_image = torch.cat([w_image] * 3, dim=0) #w_image保持单维度
        zero_input = torch.zeros_like(wr_image)
        wr_image = torch.cat([wr_image] * 3, dim=0)
        
        wr_image = torch.cat([wr_image,zero_input] , dim=0)#channel = 4
        return list(self.w_image_dict.keys())[index],w_image.float(), w_mask.float(),wr_image.float(), wr_mask.float()

    def get_patches(self, imgname,image, mask):
        # print(image.shape)
        # Split image and mask into patches using sliding window approach
        height, width = image.shape
        patch_height, patch_width = self.split_patch_size
        stride_height, stride_width = self.stride

        # Calculate the number of patches in each dimension

        num_patches_height = (height - patch_height) // stride_height + 2
        num_patches_width = (width - patch_width) // stride_width + 2
        # print(num_patches_height,num_patches_width)
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
                if self.mode=="train" and patch_mask.sum()==0:
                    pass
                else:
                # print(f"{imgname}_{start_x}_{start_y}")
                    patches_image[f"{imgname}_{start_x}_{start_y}"]= patch_image
                    patches_mask[f"{imgname}_{start_x}_{start_y}"]= patch_mask

        return patches_image, patches_mask
    

    def load_data(self, image_path, mask_path):
        # Load image and mask data using PIL
        image_data = np.array(Image.open(image_path))
        mask_data = np.array(Image.open(mask_path))
        # print(image_path,mask_path)
        # print(image_data.shape,mask_data.shape)
        assert image_data.shape==mask_data.shape
        
        return image_data, mask_data




class Fine_SAR_Water_Dataset(Dataset):
    def __init__(self, mode,imgname,w_image,w_mask,w_pred,patch_size=(64, 64), stride=(32, 32)):
        super(Fine_SAR_Water_Dataset, self).__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.image_dict = {}
        self.mask_dict = {}
        self.pred_dict = {}
        self.mode=mode
        self.train_transforms = Compose(
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
        self.test_transforms = Compose(
        [
            # EnsureChannelFirstd(keys=["image", "label"]),
            # NormalizeIntensityd(keys=["image"]),
            EnsureTyped(keys=["image", "label"]),
        ]
    )
        patches_image, patches_mask,patches_pred= self.get_patches(imgname,w_image,w_mask,w_pred)
        self.image_dict.update(patches_image)
        self.mask_dict.update(patches_mask)
        self.pred_dict.update(patches_pred)

        # print(f"{mode}, casename:{imgname},split_num:{len(patches_image)}")
        
       
        if mode =="train":
            dd = list(self.image_dict.items())
            # print(f"total_number:{len(dd)}")
            random.shuffle(dd)
            self.image_dict = dict(dd)
        
    def __len__(self):
        return len(self.image_dict)

    def __getitem__(self, index):
        # pdb.set_trace()
        image_patch = self.image_dict[list(self.image_dict.keys())[index]]
        # image_patch = torch.cat([image_patch] * 3, dim=0)
        mask_patch = self.mask_dict[list(self.image_dict.keys())[index]]
        pred_patch = self.pred_dict[list(self.image_dict.keys())[index]]
       
        ll = torch.cat([image_patch,pred_patch] , dim=0)
        # print(ll.shape)
        if self.mode=="train":
        
            train_pair = {"image":ll,"label":mask_patch}
            # Apply transformations to image and mask patches
            patch_pair =self.train_transforms(train_pair)
        else:
            test_pair = {"image":ll,"label":mask_patch}
            
            # Apply transformations to image and mask patches
            patch_pair = self.test_transforms(test_pair)
        image_patch, mask_patch = patch_pair["image"],patch_pair["label"]
       
        return list(self.image_dict.keys())[index],image_patch.float(), mask_patch.float()
           
    # def load_data(self, image_path, mask_path):
    #     # Load image and mask data using PIL
    #     image_data = np.array(Image.open(image_path))
    #     mask_data = np.array(Image.open(mask_path))
    #     # print(image_path,mask_path)
    #     # print(image_data.shape,mask_data.shape)
    #     assert image_data.shape==mask_data.shape
        
    #     return image_data, mask_data
    
    def get_patches(self, imgname,w_image,w_mask,w_pred):
        """
        @description  :get patches from image,mask and prediction(sliding-window)
        ---------
        @param  :
        -------
        @Returns  :
        -------
        """
        
        # print(image.shape)
        # Split image and mask into patches using sliding window approach
        
        patches_image = {}
        patches_mask = {}
        patches_pred = {}
        batch,c,height, width = w_image.shape
        for i in range(len(imgname)):
            image = w_image[i]
            pred = w_pred[i]
            mask = w_mask[i]
           
            name = imgname[i][:7]
           
            sx,sy = int(imgname[i].split("_")[-2]),int(imgname[i].split("_")[-1])
            patch_height, patch_width = self.patch_size
            stride_height, stride_width = self.stride

            # Calculate the number of patches in each dimension

            num_patches_height = (height - patch_height) // stride_height + 2
            num_patches_width = (width - patch_width) // stride_width + 2
            
            # Initialize the patches array with zeros
            # ll = []
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
                    patch_image = image[:,start_x:end_x, start_y:end_y]
                    patch_mask = mask[:,start_x:end_x, start_y:end_y]
                    patch_pred = pred[:,start_x:end_x, start_y:end_y]
                   
                   
                    patches_image[f"{name}_{start_x+sx}_{start_y+sy}"]= patch_image
                    patches_mask[f"{name}_{start_x+sx}_{start_y+sy}"]= patch_mask
                    patches_pred[f"{name}_{start_x+sx}_{start_y+sy}"]= patch_pred
            
            # print(f"{ kk},{len(ll)}")
        
        # print(f"fine,patch_name:{imgname},{len(patches_image)}")           
        return patches_image, patches_mask,patches_pred

    
if __name__ =="__main__":

    image_path = "Dataset/PCA_SWM/waterdata"
    mask_path = "Dataset/PCA_SWM/GT"
    image_paths = sorted(glob(os.path.join(image_path,"*.jpg"))[:1])
    mask_paths = sorted(glob(os.path.join(mask_path,"*binary.tif"))[:1])
    # image_data = np.array(Image.open(mask_paths[0]))
    # print(image_data.shape)
    # print(len(image_paths),len(mask_paths))
    SARtrain_dataset = Coarse_SAR_Water_Dataset("train","train",data_split= "SAR_water_1.pickle",split_patch_size= (512,512),patch_size=(128, 128), stride=(128,128))
    dataloader = DataLoader(SARtrain_dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=list_data_collate)


    from models.wingnet import Wingnet_encoder,Wingnet_decoder
    from models.CToF import SegmentationModel
    from torchvision import transforms
    from torch.nn import DataParallel
    encoder = Wingnet_encoder(in_channel=4,n_classes=1)
    c_decoder = Wingnet_decoder(n_classes=1)
    f_decoder = Wingnet_decoder(n_classes=1)
    coarse_model = SegmentationModel(encoder=encoder,decoder=c_decoder)
    CUDA_VISIBLE_DEVICES=2,3
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    coarse_model = DataParallel(coarse_model)
    coarse_model.to(device)
    coarse_model.eval()
    # pdb.set_trace()
    for i,(name,w_image, w_mask,wr_image, wr_mask) in enumerate(dataloader):
        print(wr_mask.shape,wr_image.shape)
        inputs = wr_image.to(device)
        labels = wr_mask.to(device)
        w_image = w_image.to(device)
        w_mask = w_mask.to(device)
        outputs = coarse_model(wr_image)
        w_pred = outputs[0]
        resize = transforms.Resize([w_mask.shape[-2],w_mask.shape[-1]])
        w_pred = resize(w_pred)
        # print(name)
        # w_pred = w_mask
        patchSARtrain_dataset = Fine_SAR_Water_Dataset( "train",name,w_image,w_mask,w_pred,patch_size=(128, 128), stride=(64, 64))
        patchdataloader = DataLoader(patchSARtrain_dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=list_data_collate)
        for j,(patch_name,patch_image,patch_mask) in enumerate(patchdataloader):
            # print(name)
            print(patch_image.shape)
            print(patch_mask.shape)
        