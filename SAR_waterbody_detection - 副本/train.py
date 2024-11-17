


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
from dataloader import SAR_Water_Dataset
from torch.utils.tensorboard import SummaryWriter
import segmentation_models_pytorch as smp
from monai.losses import DiceFocalLoss
from timm.models import create_model
from models.wingnet import WingsNet,DeepdenseWing
from torch.nn import DataParallel
from test import val
from option import parser
import torch.optim.lr_scheduler as lr_scheduler
from test import calculate_metrics

from loss import focal_loss

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
    train_dataset = SAR_Water_Dataset(mode = "train",data_split=args.dataset_split,patch_size=args.cubesize, stride=args.stride)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,num_workers=args.workers,
        pin_memory=True)
    test_dataset = SAR_Water_Dataset(mode = "test",data_split=args.dataset_split,patch_size=args.cubesize, stride=args.stride)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,num_workers=args.workers,
        pin_memory=True)
    if args.resume:
        
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
        else:
            ValueError("invalid model")
        state_dict = torch.load(args.resume)["state_dict"]
        model.load_state_dict(state_dict, strict=True)
    else:
        if args.model=="unet":

        # Load pre-trained U-Net model with Imagenet weights
            model = smp.Unet(encoder_name='resnet34', encoder_weights='imagenet', classes=1, activation="sigmoid")
        elif args.model=="unet++":
            model = smp.UnetPlusPlus(encoder_name='resnet34', encoder_weights='imagenet',  classes=1, activation="sigmoid")
        elif args.model=="deeplabv3+":
            model = smp.DeepLabV3Plus(encoder_name='resnet50',  encoder_weights='imagenet', classes=1, activation="sigmoid")
        elif args.model=="pspnet":
            model = smp.PSPNet(encoder_name='resnet50',  encoder_weights='imagenet', classes=1, activation="sigmoid")
        elif args.model=="wingnet":
            model = WingsNet()
        elif args.model=="Deepwingnet":
            model = DeepdenseWing()
        else:
            ValueError("invalid model")
    print(
        "Number of network parameters:",
        sum(param.numel() for param in model.parameters()),
    )
   
    model = DataParallel(model)
    model.to(device)
    
    # # Define the Swin Transformer model with pretrained weight
    # model_name = 'swin_base_patch4_window12_384'
    # num_classes = 1000
    # model = create_model(model_name, pretrained=True, num_classes=num_classes)
    # Define the loss function and optimizer
    criterion = DiceFocalLoss( gamma=0.75, squared_pred=True)
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Fine-tune the model
    print("---------------------Start training---------------------")
    val_dice_best = 0
   
    for epoch in range(args.start_epoch,args.epochs+1):
        train_losslist = []
        DSC_list = []
        FPR_list = []
        recall_list = []
        Precision_list = []
        model.train()
        for i, (patch_name,inputs, labels) in enumerate(tqdm(train_loader)):
            # Zero the parameter gradients
            optimizer.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)
            # Forward + backward + optimize
            outputs = model(inputs)
            # pred = outputs.cpu().data.numpy()
            # y_true = labels.cpu().data.numpy()
            if args.model=="wingnet":
                loss = criterion(outputs[0], labels) +criterion(outputs[1], labels)
            elif args.model=="Deepwingnet":
                # loss1 = criterion(outputs[0], labels) +criterion(outputs[1], labels)
                loss1 = criterion(outputs[0], labels) 
                loss2 = focal_loss(outputs[1][0], Deep_Dense_Modeling_Using_GTlabels(labels, D=2))
                loss3 = focal_loss(outputs[1][1], Deep_Dense_Modeling_Using_GTlabels(labels, D=4))
                loss4 = focal_loss(outputs[1][2], Deep_Dense_Modeling_Using_GTlabels(labels, D=8))
                # loss5 = crit(casePred3, idtm)
                loss = loss1 +  loss2 + loss3 + loss4
            else:
                loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            for j in range(inputs.shape[0]):
                if args.model=="wingnet" or args.model=="Deepwingnet":
                    y_pred = outputs[0][j, 0]>0.5
                else:
                    y_pred = outputs[j, 0]>0.5
                
                metric = calculate_metrics(y_pred, labels[j, 0])  # 概率预测与真值的dice
                
                DSC_list.append(metric[0])
                recall_list.append(metric[1])
                FPR_list.append(metric[2])
                Precision_list.append(metric[3])
            train_losslist.append(loss.item())   
        scheduler.step()  
        #--------save checkpoint------------
        state_dict = model.module.state_dict()

        torch.save(
            {"state_dict": state_dict, "args": args},
            os.path.join(save_dir, "latest.ckpt"),
        )

        # Save the model frequently
        if epoch > 79 and epoch % args.save_freq == 0:

            torch.save(
                {"state_dict": state_dict, "args": args},
                os.path.join(save_dir, "%03d.ckpt" % epoch),
            )

        if (epoch > 130 and epoch % args.val_freq == 0) or epoch > 180:
            
            dsc_mean,recall_mean,fpr_mean,precision_mean = val(epoch,model,device, test_loader,save_dir)
            if dsc_mean > val_dice_best:
                val_dice_best = dsc_mean
                torch.save(
                    {"state_dict": state_dict, "args": args, "epoch": epoch},
                    os.path.join(save_dir, "val_dsc_best.ckpt"),
                )
                
            # Print statistics
            
        
        mean_loss = np.mean(np.array(train_losslist))
        mean_dsc = np.mean(np.array(DSC_list))
        mean_recall = np.mean(np.array(recall_list))
        mean_fpr = np.mean(np.array(FPR_list))
        mean_precision = np.mean(np.array(Precision_list))
        logger.add_scalar("LR", optimizer.state_dict()['param_groups'][0]['lr'], global_step=epoch)
        logger.add_scalar("train_loss", mean_loss, global_step=epoch)
        logger.add_scalar("train_dsc", mean_dsc, global_step=epoch)
        logger.add_scalar("train_recall", mean_recall, global_step=epoch)
        logger.add_scalar("train_precision", mean_precision, global_step=epoch)
        print(
        "%s, epoch %d, loss %.4f, DSC %.4f, Recall %.4f, FPR %.4f,Precision %.4f"
        % (
            "train",
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
