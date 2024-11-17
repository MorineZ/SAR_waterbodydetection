
import argparse

parser = argparse.ArgumentParser(description="PyTorch Airway Segmentation")


#---------------Train--------------
parser.add_argument("--model", type=str, help="model")
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument("--savepath", type=str, help="savepath")
parser.add_argument(
    "--save_freq", default="5", type=int, metavar="S", help="save frequency"
)
parser.add_argument(
    "--val_freq", default="5", type=int, metavar="S", help="validation frequency")
parser.add_argument(
    "--epochs", default=60, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "--start_epoch",
    default=1,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "-b",
    "--batch_size",
    default=4,
    type=int,
    metavar="N",
    help="mini-batch size (default: 16)",
)

parser.add_argument(
    "--cubesize",
    default=[256,256],
    nargs="*",
    type=int,
    metavar="cube",
    help="cube size",
)
parser.add_argument(
    "--stride",
    default=[128,128],
    nargs="*",
    type=int,
    metavar="cube",
    help="stride",
)
parser.add_argument(
    "--dataset_split",
    default="",
    type=str,
    metavar="SAVE",
    help="directory to save checkpoint (default: none)",
)
parser.add_argument(
    "-j",
    "--workers",
    default=8,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 8)",
)




#---------------Test--------------
parser.add_argument("--ckpath", type=str, help="checkpoint_path")
parser.add_argument("--outputpath", type=str, help="save_path for evaluation result")