


# Import necessary libraries
import torch
import torch.nn as nn
import os
import numpy as np
import random
from glob import glob
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys
import monai
from einops import rearrange
from dataloaderv2 import Coarse_SAR_Water_Dataset,Fine_SAR_Water_Dataset
from dataloader import SAR_Water_Dataset
from torch.utils.tensorboard import SummaryWriter
import segmentation_models_pytorch as smp
from monai.losses import DiceFocalLoss
from timm.models import create_model
from models.wingnet import WingsNet,DeepdenseWing
from torch.nn import DataParallel
from test import val,CToF_val
from option import parser
import torch.optim.lr_scheduler as lr_scheduler
from test import calculate_metrics
from models.CToF import SegmentationModel
from loss import focal_loss
from torchvision import transforms
from models.wingnet import Wingnet_encoder,Wingnet_decoder,DeepWingnet_decoder
import time
from test import merge_patches


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

class Logger(object):
    """
    Logger from screen to txt file
    """

    def __init__(self, logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train():
    # Create custom dataset and dataloader
    setup_seed(0)
    global args
    args = parser.parse_args()

    save_dir = args.savepath
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logfile = os.path.join(save_dir, "log.txt")
    logger = SummaryWriter(log_dir=os.path.join(save_dir, "board_log"))
    sys.stdout = Logger(logfile)
    print("---------------------Load data---------------------")
    C_train_dataset = Coarse_SAR_Water_Dataset("train","train",data_split= args.dataset_split,split_patch_size = (1024,1024),patch_size=args.cubesize,stride=(512,512))
    C_train_loader = DataLoader(C_train_dataset, batch_size=8, shuffle=True,num_workers=args.workers,
        pin_memory=True)
    C_test_dataset = Coarse_SAR_Water_Dataset("test","test",data_split= args.dataset_split,split_patch_size = (1024,1024),patch_size=args.cubesize,stride=(512,512))
    C_test_loader = DataLoader(C_test_dataset, batch_size=8, shuffle=False,num_workers=args.workers,
        pin_memory=True)
    encoder = Wingnet_encoder(in_channel=4,n_classes=1)
    c_decoder = DeepWingnet_decoder(n_classes=1)
    f_decoder = DeepWingnet_decoder(n_classes=1)
    coarse_model = SegmentationModel(encoder=encoder,decoder=c_decoder)
    fine_model = SegmentationModel(encoder=encoder,decoder=f_decoder)
    
    if args.resume:
        c_state_dict = torch.load(args.resume)["c_state_dict"]
        coarse_model.load_state_dict(c_state_dict, strict=True)
  
        f_state_dict = torch.load(args.resume)["f_state_dict"]
        fine_model.load_state_dict(f_state_dict, strict=True)
    
    print(
        "Number of network parameters:",
        sum(param.numel() for param in coarse_model.parameters()),
    )
   
    coarse_model = DataParallel(coarse_model)
    coarse_model.to(device)
    fine_model = DataParallel(fine_model)
    fine_model.to(device)
    
    # # Define the Swin Transformer model with pretrained weight
    # model_name = 'swin_base_patch4_window12_384'
    # num_classes = 1000
    # model = create_model(model_name, pretrained=True, num_classes=num_classes)
    # Define the loss function and optimizer
    criterion = DiceFocalLoss( gamma=0.75, squared_pred=True)
    
    optimizer1 = optim.Adam(coarse_model.parameters(), lr=0.01)
    optimizer2 = optim.Adam(fine_model.parameters(), lr=0.01)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer1, T_max=args.epochs)
    
    # Fine-tune the model
    print("---------------------Start training---------------------")
    val_dice_best = 0
    
   
    for epoch in range(args.start_epoch,args.epochs+1):
        train_losslist = []
        train_patchlosslist = []
        DSC_list = []
        FPR_list = []
        recall_list = []
        Precision_list = []
        coarse_model.train()
        fine_model.train()
        start_time = time.time()

    # 1.-----Coarse segmentation stage--

        for i, (img_name,w_image, w_mask,wr_image, wr_mask) in enumerate(tqdm(C_train_loader)):
            # Zero the parameter gradients
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            
            inputs = wr_image.to(device)
            labels = wr_mask.to(device)
            w_image = w_image.to(device)
            w_mask = w_mask.to(device)
            # Forward + backward + optimize
            outputs = coarse_model(inputs)

            if args.model=="wingnet":
                loss = criterion(outputs[0], labels) +criterion(outputs[1], labels)
            elif args.model=="Deepwingnet":
                loss1 = criterion(outputs[0], labels) +criterion(outputs[1], labels)
                loss2 = focal_loss(outputs[2][0], Deep_Dense_Modeling_Using_GTlabels(labels, D=2))
                loss3 = focal_loss(outputs[2][1], Deep_Dense_Modeling_Using_GTlabels(labels, D=4))
                loss4 = focal_loss(outputs[2][2], Deep_Dense_Modeling_Using_GTlabels(labels, D=8))
                # loss5 = crit(casePred3, idtm)
                loss = loss1 +  loss2 + loss3 + loss4
            else:
                loss = criterion(outputs, labels) 
            loss.backward()
            optimizer1.step()
            train_losslist.append(loss.item()) 
            
            # c_paradict = {name:param.data for name,param in coarse_model.named_parameters()}
            # print(f"encoder weight of coarse_model : {c_paradict['module.encoder.ec1.conv1.weight'][0,0]}")
            # print(f"decoder weight of coarse_model : {c_paradict['module.decoder.dc3.conv1.weight'][0,0]}")


    # 2.-----Fine segmentation stage--
        #2-1:coarse stage prediction
        coarse_model.eval()
        for i, (img_name,w_image, w_mask,wr_image, wr_mask) in enumerate(tqdm(C_test_loader)):
        
            inputs = wr_image.to(device)
            labels = wr_mask.to(device)
            w_image = w_image.to(device)
            w_mask = w_mask.to(device)
            outputs = coarse_model(inputs)
            # loss = DiceFocalLoss(to_onehot_y=True, gamma=0.75, squared_pred=True)(outputs, labels)
            if args.model=="wingnet" or args.model=="Deepwingnet":
                y_pred = outputs[0]
            else:
                y_pred = outputs
            resize = transforms.Resize([w_mask.shape[-2],w_mask.shape[-1]])
            y_pred = resize(y_pred)
                
        
        # merged_preds = merge_patches(device,all_preds,patch_shape= (patch_inputs.shape[-2],patch_inputs.shape[-1]),img_shape = (6944,6438))
        # merged_labels = merge_patches(device,all_labels,patch_shape= (patch_inputs.shape[-2],patch_inputs.shape[-1]),img_shape = (6944,6438))
            F_train_dataset = Fine_SAR_Water_Dataset("train",img_name,w_image,w_mask,y_pred,patch_size=args.cubesize, stride=args.stride)
            F_train_loader = DataLoader(F_train_dataset, batch_size=args.batch_size, shuffle=True)
            
            for j,(patch_name,patch_image,patch_mask) in enumerate(F_train_loader):
                patch_inputs = patch_image.to(device)
                patch_labels = patch_mask.to(device)
                # Forward + backward + optimize
                patch_outputs = fine_model(patch_inputs)

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
                # loss = criterion(outputs, labels)
                # loss = 
                patch_loss.backward()
                optimizer2.step()
                # c_paradict = {name:param.data for name,param in coarse_model.named_parameters()}
                # print(f"encoder weight of coarse_model : {c_paradict['module.encoder.ec1.conv1.weight'][0,0]}")
                # print(f"decoder weight of coarse_model : {c_paradict['module.decoder.dc3.conv1.weight'][0,0]}")
                # f_paradict = {name:param.data for name,param in fine_model.named_parameters()}
                # print(f"encoder weight of fine_model : {f_paradict['module.encoder.ec1.conv1.weight'][0,0]}")
                # print(f"decoder weight of fine_model : {f_paradict['module.decoder.dc3.conv1.weight'][0,0]}")

                for kk in range(patch_inputs.shape[0]):
                    if args.model=="wingnet" or args.model=="Deepwingnet":
                        y_patch_pred = patch_outputs[0][kk, 0]>0.5
                    else:
                        y_patch_pred = patch_outputs[kk, 0]>0.5
                    metric = calculate_metrics(y_patch_pred, patch_labels[kk, 0])  # 概率预测与真值的dice
                    
                    DSC_list.append(metric[0])
                    recall_list.append(metric[1])
                    FPR_list.append(metric[2])
                    Precision_list.append(metric[3])
                
                train_patchlosslist.append(patch_loss.item())
        scheduler.step()  
        end_time = time.time()
        #--------save checkpoint------------
        
        torch.save(
            {"c_state_dict": coarse_model.module.state_dict(), 
             "f_state_dict": fine_model.module.state_dict(),
             "args": args,
             "epoch": epoch},
            os.path.join(save_dir, "latest.ckpt"),
        )

        # Save the model frequently
        if epoch > 79 and epoch % args.save_freq == 0:
            torch.save(
            {"c_state_dict": coarse_model.module.state_dict(), 
             "f_state_dict": fine_model.module.state_dict(),
             "args": args,
             "epoch": epoch},
            os.path.join(save_dir, "%03d.ckpt" % epoch),
        )


        if (epoch > 130 and epoch % args.val_freq == 0) or epoch > 180:
            
            dsc_mean,recall_mean,fpr_mean,precision_mean = CToF_val(epoch,coarse_model,fine_model,device, C_test_loader,save_dir)
            if dsc_mean > val_dice_best:
                val_dice_best = dsc_mean
                torch.save(
                {"c_state_dict": coarse_model.module.state_dict(), 
                "f_state_dict": fine_model.module.state_dict(),
                "args": args,
             "epoch": epoch},
                os.path.join(save_dir, "val_dsc_best.ckpt"))
                
                
            # Print statistics
            
        
        mean_loss = np.mean(np.array(train_patchlosslist))
        mean_dsc = np.mean(np.array(DSC_list))
        mean_recall = np.mean(np.array(recall_list))
        mean_fpr = np.mean(np.array(FPR_list))
        mean_precision = np.mean(np.array(Precision_list))
        logger.add_scalar("LR", optimizer1.state_dict()['param_groups'][0]['lr'], global_step=epoch)
        logger.add_scalar("train_loss", mean_loss, global_step=epoch)
        logger.add_scalar("train_dsc", mean_dsc, global_step=epoch)
        logger.add_scalar("train_recall", mean_recall, global_step=epoch)
        logger.add_scalar("train_precision", mean_precision, global_step=epoch)
        print(
        "%s, time %.4f,epoch %d, loss %.4f, DSC %.4f, Recall %.4f, FPR %.4f,Precision %.4f"
        % (
            "train",
            end_time-start_time,
            epoch,
            mean_loss,
            mean_dsc,
            mean_recall,
            mean_fpr,
            mean_precision
            )   
        )

    print('Finished training')
    val_dsc_best_epoch = torch.load(os.path.join(save_dir, "val_dsc_best.ckpt"))[
        "epoch"
    ]
    os.rename(
        os.path.join(save_dir, "val_dsc_best.ckpt"),
        os.path.join(save_dir, "val_dsc_best_%03d.ckpt" % val_dsc_best_epoch),
    )
    print(f"val_dsc_best_epoch:{val_dsc_best_epoch}")



if __name__=="__main__":

    train()
