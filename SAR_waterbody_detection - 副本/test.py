



import torch
import numpy as np
from tqdm import tqdm
import segmentation_models_pytorch as smp
from PIL import Image
import pandas as pd
from option import parser
import os
from einops import rearrange
from torch.nn import DataParallel
from dataloaderv2 import Coarse_SAR_Water_Dataset,Fine_SAR_Water_Dataset
import cv2
from models.wingnet import WingsNet,DeepdenseWing
from dataloader import SAR_Water_Dataset
from torch.utils.data import Dataset, DataLoader
from monai.losses import DiceFocalLoss
from torchvision import transforms
from loss import focal_loss
import pdb
def Deep_Dense_Modeling_Using_GTlabels(GT_labels, D=4, reverse=False):
    """
    @description  : Reproduction vision -> AirwayNet: A Voxel-Connectivity Aware Approach for Accurate Airway Segmentation Using Convolutional Neural Networks, MICCAI2019
    ---------
    @param  :   GT_labels -> binary Ground Truth masks _numpy_array
                D-> downsampling ratio
    -------
    @Returns
    -------
    """
    if reverse:
        GT_Up_labels = rearrange(
            GT_labels,
            " b (c ph pw ) h w  -> b c (h ph) (w pw)",
            ph=D,
            pw=D,
        )
        Tf_label = GT_Up_labels
    else:
        GT_Down_labels = rearrange(
            GT_labels,
            "b c (h ph) (w pw)-> b (c ph pw) h w",
            ph=D,
            pw=D,       
        )
        Tf_label = GT_Down_labels
    return Tf_label


def merge_patches(device,patches_pred,patch_shape,img_shape = (6944,6438)):
    
    height, width = img_shape
    patch_height, patch_width = patch_shape
   
    preds = {}
    counts = {}
    for i in patches_pred.keys():
    
        casename = i[:7]
        if casename not in preds.keys():
            preds[casename] = torch.zeros((height, width),dtype=torch.float32).to(device)
            counts[casename] = torch.zeros((height, width),dtype=torch.float32).to(device)
        patch_pred = patches_pred[i]
        # print(patch_pred.dtype)
        start_x = int(i.split("_")[-2])
        start_y = int(i.split("_")[-1])
        if start_x>6000 and start_y>6000:
            print(start_x,start_y)
       
        preds[casename][start_x:start_x+patch_height, start_y:start_y+patch_width] += patch_pred
        
        counts[casename][start_x:start_x+patch_height, start_y:start_y+patch_width] += 1
        
    
    for casename in preds.keys():

        preds[casename] = preds[casename]/counts[casename]
      
        # pdb.set_trace()
    return preds

def calculate_metrics(pred, target):
    """
    Calculates evaluation metrics including DSC, FPR, Precision, and Recall.
    Assumes pred and target are both torch.Tensor objects.
    """
    smooth=1.0  # small value to avoid division by zero

    # Convert tensors to numpy arrays
    # pred = pred.detach().cpu().numpy()
    # target = target.detach().cpu().numpy()

    # Calculate true positives, false positives, and false negatives
    y_true_f = target.flatten()
    y_pred_f = pred.flatten()
    intersection = torch.sum(y_true_f * y_pred_f)
    dsc =  (2.0 * intersection + smooth) / (
        torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth
    )


    recall =  torch.sum(y_true_f * y_pred_f) / (torch.sum(y_true_f) + smooth)


    fpr = torch.sum((1.0 - y_true_f) * y_pred_f) / (
        torch.sum((1.0 - y_true_f)) + smooth
    )

    tp = torch.sum(y_pred_f * y_true_f) + smooth
    precision = tp / (torch.sum(y_pred_f) + smooth)
    
    return dsc.item(), recall.item(),fpr.item(), precision.item()


def val(epoch,model,device, test_loader,save_dir):
    global args
    args = parser.parse_args()
    criterion = DiceFocalLoss( gamma=0.75, squared_pred=True)
    metrics_csv = {
        "name": [],
        "DSC": [],
        "Recall": [],
        "FPR": [],
        "Precision": [],
        
    }
    losslist = []
    model.eval()
    with torch.no_grad():
        all_preds = {}
        all_labels = {}
        for patch_name,inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            # loss = DiceFocalLoss(to_onehot_y=True, gamma=0.75, squared_pred=True)(outputs, labels)
            if args.model=="wingnet":
                loss = criterion(outputs[0], labels) +criterion(outputs[1], labels)
            elif args.model=="Deepwingnet":
                loss1 = criterion(outputs[0], labels) 
                loss2 = focal_loss(outputs[1][0], Deep_Dense_Modeling_Using_GTlabels(labels, D=2))
                loss3 = focal_loss(outputs[1][1], Deep_Dense_Modeling_Using_GTlabels(labels, D=4))
                loss4 = focal_loss(outputs[1][2], Deep_Dense_Modeling_Using_GTlabels(labels, D=8))
                # loss5 = crit(casePred3, idtm)
                loss = loss1 +  loss2 + loss3 + loss4
            else:
                loss = criterion(outputs, labels) 
            for j in range(inputs.shape[0]):
                if args.model=="wingnet" or args.model=="Deepwingnet":
                    y_pred = outputs[0][j, 0]>0.5
                else:
                    y_pred = outputs[j, 0]>0.5

                # metric = calculate_metrics(y_pred, labels[j, 0])
                all_preds[patch_name[j]] = y_pred
                all_labels[patch_name[j]] = labels[j, 0]
                # metrics_csv["name"].append(patch_name[j])
                # metrics_csv["DSC"].append(metric[0])
                # metrics_csv["Recall"].append(metric[1])
                # metrics_csv["FPR"].append(metric[2])
                # metrics_csv["Precision"].append(metric[3])
        
            losslist.append(loss.item())
        # # Merge the predictions and labels
        
        merged_preds = merge_patches(device,all_preds,patch_shape= (inputs.shape[-2],inputs.shape[-1]),img_shape = (6944,6438))
        merged_labels = merge_patches(device,all_labels,patch_shape= (inputs.shape[-2],inputs.shape[-1]),img_shape = (6944,6438))
        
        for casename in merged_preds.keys():
            # Calculate the metrics on the merged prediction and label
            
            metric = calculate_metrics(merged_preds[casename], merged_labels[casename])
            # print(metric)
            metrics_csv["name"].append(casename)
            metrics_csv["DSC"].append(metric[0])
            metrics_csv["Recall"].append(metric[1])
            metrics_csv["FPR"].append(metric[2])
            metrics_csv["Precision"].append(metric[3])
        
    meanloss = np.mean(np.array(losslist))
    dsc_mean,recall_mean,fpr_mean,precision_mean = np.mean(np.array(metrics_csv["DSC"])),np.mean(np.array(metrics_csv["Recall"])),np.mean(np.array(metrics_csv["FPR"])),np.mean(np.array(metrics_csv["Precision"]))
    metrics_csv["name"].append("Mean")
    metrics_csv["DSC"].append(dsc_mean)
    metrics_csv["Recall"].append(recall_mean)
    metrics_csv["FPR"].append(fpr_mean)
    metrics_csv["Precision"].append(precision_mean)

    metrics_csv["name"].append("Std")
    metrics_csv["DSC"].append(np.std(np.array(metrics_csv["DSC"])[:-1]))
    metrics_csv["Recall"].append(np.std(np.array(metrics_csv["Recall"])[:-1]))
    metrics_csv["FPR"].append(np.std(np.array(metrics_csv["FPR"])[:-1]))
    metrics_csv["Precision"].append(np.std(np.array(metrics_csv["Precision"])[:-1]))

    df = pd.DataFrame(metrics_csv)
    csv_path = os.path.join(save_dir, "test_%03d.csv" % epoch)
    df.to_csv(csv_path)
    print(
    "%s, epoch %d, loss %.4f, DSC %.4f, Recall %.4f, FPR %.4f,Precision %.4f"
    % (
        "val",
        epoch,
        meanloss,
        dsc_mean,
        recall_mean,
        fpr_mean,
        precision_mean
        )   
    )
    return dsc_mean,recall_mean,fpr_mean,precision_mean


def deploy():

    global args
    args = parser.parse_args()
    save_dir = args.savepath
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
#------1.data_load-------------
    test_dataset = SAR_Water_Dataset(mode = "test",data_split=args.dataset_split,patch_size=args.cubesize, stride=args.stride)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,num_workers=args.workers,
        pin_memory=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#-----2.model_setting----------------
    if args.model=="unet":

    # Load pre-trained U-Net model with Imagenet weights
        model = smp.Unet(encoder_name='resnet34', encoder_weights=None, classes=1, activation="sigmoid")
    elif args.model=="unet++":
        model = smp.UnetPlusPlus(encoder_name='resnet34', encoder_weights=None,  classes=1, activation="sigmoid")
    elif args.model=="deeplabv3+":
        model = smp.DeepLabV3Plus(encoder_name='resnet50',  encoder_weights=None, classes=1, activation="sigmoid")
    elif args.model=="pspnet":
        model = smp.PSPNet(encoder_name='resnet50',  encoder_weights=None, classes=1, activation="sigmoid")
    elif args.model=="wingnet":
        model = WingsNet()
    elif args.model=="Deepwingnet":
        model = DeepdenseWing()
    else:
        ValueError("invalid model")
    state_dict = torch.load(args.resume , weights_only =False)
    model.load_state_dict(state_dict, strict=False)
    model = DataParallel(model)
    model.to(device)
    metrics_csv = {
        "name": [],
        "DSC": [],
        "Recall": [],
        "FPR": [],
        "Precision": [],
        
    }
    # losslist = []

    model.eval()
    with torch.no_grad():
        all_preds = {}
        all_labels = {}
        for patch_name,inputs, labels in tqdm(test_loader):
            inputs = inputs.to(device)
            # print(inputs.shape)
            labels = labels.to(device)
            outputs = model(inputs)
            # loss = DiceFocalLoss(to_onehot_y=True, gamma=0.75, squared_pred=True)(outputs[0], labels)+\
            # DiceFocalLoss(to_onehot_y=True, gamma=0.75, squared_pred=True)(outputs[1], labels)
            for j in range(inputs.shape[0]):
                if args.model=="wingnet" or args.model=="Deepwingnet":
                    y_pred = outputs[0][j, 0]>0.5
                else:
                    y_pred = outputs[j, 0]>0.5
                # metric = calculate_metrics(y_pred, labels[j, 0])
                all_preds[patch_name[j]] = y_pred
                all_labels[patch_name[j]] = labels[j, 0]
                # metrics_csv["name"].append(patch_name[j])
                # metrics_csv["DSC"].append(metric[0])
                # metrics_csv["Recall"].append(metric[1])
                # metrics_csv["FPR"].append(metric[2])
                # metrics_csv["Precision"].append(metric[3])
        
            # losslist.append(loss.item())
        # # Merge the predictions and labels
        merged_preds = merge_patches(device,all_preds,patch_shape= (inputs.shape[-2],inputs.shape[-1]),img_shape = (6944,6438))
        merged_labels = merge_patches(device,all_labels,patch_shape= (inputs.shape[-2],inputs.shape[-1]),img_shape = (6944,6438))
        for casename in merged_preds.keys():
            # Calculate the metrics on the merged prediction and label
            metric = calculate_metrics(merged_preds[casename], merged_labels[casename])
            # print(metric)
            metrics_csv["name"].append(casename)
            metrics_csv["DSC"].append(metric[0])
            metrics_csv["Recall"].append(metric[1])
            metrics_csv["FPR"].append(metric[2])
            metrics_csv["Precision"].append(metric[3])
        
    #-------------save-merged-images--------------
    print(merged_preds.keys())
    for casename in merged_preds.keys():
        pred_ = merged_preds[casename].cpu().numpy()
        pred = pred_ *255
        cv2.imwrite(os.path.join(save_dir,casename+"_pred.png"),pred)
        pred_binary =  np.int32(pred_>0.5)*255
        cv2.imwrite(os.path.join(save_dir,casename+"_pred_binary.png"),pred_binary)
    dsc_mean,recall_mean,fpr_mean,precision_mean = np.mean(np.array(metrics_csv["DSC"])),np.mean(np.array(metrics_csv["Recall"])),np.mean(np.array(metrics_csv["FPR"])),np.mean(np.array(metrics_csv["Precision"]))
    metrics_csv["name"].append("Mean")
    metrics_csv["DSC"].append(dsc_mean)
    metrics_csv["Recall"].append(recall_mean)
    metrics_csv["FPR"].append(fpr_mean)
    metrics_csv["Precision"].append(precision_mean)

    metrics_csv["name"].append("Std")
    metrics_csv["DSC"].append(np.std(np.array(metrics_csv["DSC"])[:-1]))
    metrics_csv["Recall"].append(np.std(np.array(metrics_csv["Recall"])[:-1]))
    metrics_csv["FPR"].append(np.std(np.array(metrics_csv["FPR"])[:-1]))
    metrics_csv["Precision"].append(np.std(np.array(metrics_csv["Precision"])[:-1]))

    df = pd.DataFrame(metrics_csv)
    csv_path = os.path.join(save_dir, "test_result.csv" )
    df.to_csv(csv_path)
    return None


def CToF_val(epoch,C_model,F_model,device, C_test_loader,save_dir):
    global args
    args = parser.parse_args()
    criterion = DiceFocalLoss( gamma=0.75, squared_pred=True)
    metrics_csv = {
        "name": [],
        "DSC": [],
        "Recall": [],
        "FPR": [],
        "Precision": [],
        
    }
    losslist = []
    C_model.eval()
    F_model.eval()
    with torch.no_grad():
        all_preds = {}
        all_labels = {}
        for i, (img_name,w_image, w_mask,wr_image, wr_mask) in enumerate(tqdm(C_test_loader)):
           
            inputs = wr_image.to(device)
            labels = wr_mask.to(device)
            w_image = w_image.to(device)
            w_mask = w_mask.to(device)
            outputs = C_model(inputs)
            # loss = DiceFocalLoss(to_onehot_y=True, gamma=0.75, squared_pred=True)(outputs, labels)
            if args.model=="wingnet" or args.model=="Deepwingnet":
                y_pred = outputs[0]
            else:
                y_pred = outputs
            resize = transforms.Resize([w_mask.shape[-2],w_mask.shape[-1]])
            y_pred = resize(y_pred)
            F_test_dataset = Fine_SAR_Water_Dataset("test",img_name,w_image,w_mask,y_pred,patch_size=args.cubesize, stride=args.stride)
            F_test_loader = DataLoader(F_test_dataset, batch_size=args.batch_size, shuffle=False)
            for j,(patch_name,patch_image,patch_mask) in enumerate(F_test_loader):
                patch_inputs = patch_image.to(device)
                patch_labels = patch_mask.to(device)
                patch_outputs = F_model(patch_inputs)
                if args.model=="wingnet":
                    patch_loss = criterion(patch_outputs[0], patch_labels) +criterion(patch_outputs[1], patch_labels)
                elif args.model=="Deepwingnet":
                    loss1 = criterion(patch_outputs[0], patch_labels) +criterion(patch_outputs[1], patch_labels)
                    loss2 = focal_loss(patch_outputs[2][0], Deep_Dense_Modeling_Using_GTlabels(patch_labels, D=2))
                    loss3 = focal_loss(patch_outputs[2][1], Deep_Dense_Modeling_Using_GTlabels(patch_labels, D=4))
                    loss4 = focal_loss(patch_outputs[2][2], Deep_Dense_Modeling_Using_GTlabels(patch_labels, D=8))
                    # loss5 = crit(casePred3, idtm)
                    patch_loss = loss1 +  loss2 + loss3 + loss4
                else:
                    patch_loss = criterion(patch_outputs, patch_labels)
                for kk in range(patch_inputs.shape[0]):
                   
                    if args.model=="wingnet" or args.model=="Deepwingnet":
                        y_patch_pred = patch_outputs[0][kk, 0]>0.5
                    else:
                        y_patch_pred = patch_outputs[kk, 0]>0.5
                    all_preds[patch_name[kk]] = y_patch_pred
                    all_labels[patch_name[kk]] = patch_labels[kk, 0]
                # metric = calculate_metrics(y_pred, labels[j, 0])
                # print(sum(y_patch_pred))
                
                # metrics_csv["name"].append(patch_name[j])
                # metrics_csv["DSC"].append(metric[0])
                # metrics_csv["Recall"].append(metric[1])
                # metrics_csv["FPR"].append(metric[2])
                # metrics_csv["Precision"].append(metric[3])
        
                losslist.append(patch_loss.item())
        # # Merge the predictions and labels
        
        merged_preds = merge_patches(device,all_preds,patch_shape= (patch_inputs.shape[-2],patch_inputs.shape[-1]),img_shape = (6944,6438))
        merged_labels = merge_patches(device,all_labels,patch_shape= (patch_inputs.shape[-2],patch_inputs.shape[-1]),img_shape = (6944,6438))
        # print(sum(merged_preds["10473_4"]))
        for casename in merged_preds.keys():
            # Calculate the metrics on the merged prediction and label
            metric = calculate_metrics(merged_preds[casename], merged_labels[casename])
            # print(metric)
            metrics_csv["name"].append(casename)
            metrics_csv["DSC"].append(metric[0])
            metrics_csv["Recall"].append(metric[1])
            metrics_csv["FPR"].append(metric[2])
            metrics_csv["Precision"].append(metric[3])
        
    meanloss = np.mean(np.array(losslist))
    dsc_mean,recall_mean,fpr_mean,precision_mean = np.mean(np.array(metrics_csv["DSC"])),np.mean(np.array(metrics_csv["Recall"])),np.mean(np.array(metrics_csv["FPR"])),np.mean(np.array(metrics_csv["Precision"]))
    metrics_csv["name"].append("Mean")
    metrics_csv["DSC"].append(dsc_mean)
    metrics_csv["Recall"].append(recall_mean)
    metrics_csv["FPR"].append(fpr_mean)
    metrics_csv["Precision"].append(precision_mean)

    metrics_csv["name"].append("Std")
    metrics_csv["DSC"].append(np.std(np.array(metrics_csv["DSC"])[:-1]))
    metrics_csv["Recall"].append(np.std(np.array(metrics_csv["Recall"])[:-1]))
    metrics_csv["FPR"].append(np.std(np.array(metrics_csv["FPR"])[:-1]))
    metrics_csv["Precision"].append(np.std(np.array(metrics_csv["Precision"])[:-1]))

    df = pd.DataFrame(metrics_csv)
    csv_path = os.path.join(save_dir, "test_%03d.csv" % epoch)
    df.to_csv(csv_path)
    print(
    "%s, epoch %d, loss %.4f, DSC %.4f, Recall %.4f, FPR %.4f,Precision %.4f"
    % (
        "val",
        epoch,
        meanloss,
        dsc_mean,
        recall_mean,
        fpr_mean,
        precision_mean
        )   
    )
    return dsc_mean,recall_mean,fpr_mean,precision_mean       
if __name__=="__main__":
    deploy()