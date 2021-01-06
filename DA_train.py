import argparse
import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import os
import os.path as osp
import random

from model.deeplab_multi import DeeplabMulti
from model.discriminator import FCDiscriminator
from model.discriminator import OutspaceDiscriminator
from dataset.gta5_dataset import GTA5DataSet
from dataset.cityscapes_dataset import cityscapesDataSet

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

MODEL = 'DeepLab'
BATCH_SIZE = 1
NUM_WORKERS = 8
IGNORE_LABEL = 255
LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
NUM_CLASSES = 19
NUM_STEPS = 250000
NUM_STEPS_STOP = 150000  # early stopping
POWER = 0.9
RESTORE_FROM = './model_weight/DeepLab_resnet_pretrained_init-f81d91e8.pth'
SAVE_PRED_EVERY = 2000
SNAPSHOT_DIR = './snapshots/'
WEIGHT_DECAY = 0.0005

LEARNING_RATE_D = 1e-4

SOURCE = 'GTA5'
INPUT_SIZE = '1280,720'
DATA_DIRECTORY = './data/GTA5/'
DATA_LIST_PATH = './dataset/gta5_list/train.txt'

TARGET = 'cityscapes'
INPUT_SIZE_TARGET = '1024,512'
DATA_DIRECTORY_TARGET = './data/Cityscapes/'
DATA_LIST_PATH_TARGET = './dataset/cityscapes_list/train.txt'
SET = 'train'


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab")
    parser.add_argument("--target", type=str, default=TARGET,
                        help="available options : cityscapes")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument("--data-dir-target", type=str, default=DATA_DIRECTORY_TARGET,
                        help="Path to the directory containing the target dataset.")
    parser.add_argument("--data-list-target", type=str, default=DATA_LIST_PATH_TARGET,
                        help="Path to the file listing the images in the target dataset.")
    parser.add_argument("--input-size-target", type=str, default=INPUT_SIZE_TARGET,
                        help="Comma-separated string with height and width of target images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP,
                        help="Number of training steps for early stopping.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--cpu", action='store_true', help="choose to use cpu device.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose adaptation set.")
    return parser.parse_args()


args = get_arguments()


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def adjust_learning_rate_D(optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
def main():
    """Create the model and start the training."""
    setup_seed(666)
    device = torch.device("cuda" if not args.cpu else "cpu")

    w, h = map(int, args.input_size.split(','))
    input_size = (w, h)

    w, h = map(int, args.input_size_target.split(','))
    input_size_target = (w, h)

    cudnn.enabled = True

    # Create network
    model = DeeplabMulti(num_classes=args.num_classes)
    saved_state_dict = torch.load(args.restore_from)
    new_params = model.state_dict().copy()
    for i in saved_state_dict:
        i_parts = i.split('.')
        if not args.num_classes == 19 or not i_parts[1] == 'layer5':
            if i_parts[1]=='layer4' and i_parts[2]=='2':
                i_parts[1] = 'layer5'
                i_parts[2] = '0'
                new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
            else:
                new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
    model.load_state_dict(new_params)
    model.train()
    model.to(device)

    cudnn.benchmark = True

    # init D
    num_class_list = [2048, 19]
    model_D = nn.ModuleList([FCDiscriminator(num_classes=num_class_list[i]).train().to(device) if i<1 else OutspaceDiscriminator(num_classes=num_class_list[i]).train().to(device) for i in range(2)])
    
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    trainloader = data.DataLoader(
        GTA5DataSet(args.data_dir, args.data_list, max_iters=args.num_steps * args.batch_size,
                    crop_size=input_size,
                    scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    trainloader_iter = enumerate(trainloader)

    targetloader = data.DataLoader(cityscapesDataSet(args.data_dir_target, args.data_list_target,
                                                     max_iters=args.num_steps * args.batch_size,
                                                     crop_size=input_size_target,
                                                     scale=False, mirror=args.random_mirror, mean=IMG_MEAN,
                                                     set=args.set),
                                   batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                   pin_memory=True)


    targetloader_iter = enumerate(targetloader)

    # implement model.optim_parameters(args) to handle different models' lr setting

    optimizer = optim.SGD(model.optim_parameters(args),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()

    optimizer_D = optim.Adam(model_D.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D.zero_grad()

    bce_loss = torch.nn.MSELoss()
    seg_loss = torch.nn.CrossEntropyLoss(ignore_index=255)

    interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear', align_corners=True)
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear', align_corners=True)

    # labels for adversarial training
    source_label = 0
    target_label = 1

    for i_iter in range(args.num_steps):

        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter)

        optimizer_D.zero_grad()
        adjust_learning_rate_D(optimizer_D, i_iter)

        # train G
        # don't accumulate grads in D
        for param in model_D.parameters():
            param.requires_grad = False

        # train with source
        _, batch = trainloader_iter.__next__()
        images, labels, _, _ = batch
        images = images.to(device)
        labels = labels.long().to(device)

        feat_source, pred_source = model(images, model_D, 'source')
        pred_source = interp(pred_source)

        loss_seg = seg_loss(pred_source, labels)
        loss_seg.backward()

        # train with target
        _, batch = targetloader_iter.__next__()
        images, _, _ = batch
        images = images.to(device)

        feat_target, pred_target = model(images, model_D, 'target')
        pred_target = interp_target(pred_target)

        loss_adv = 0
        D_out = model_D[0](feat_target)
        loss_adv += bce_loss(D_out, torch.FloatTensor(D_out.data.size()).fill_(source_label).to(device))
        D_out = model_D[1](F.softmax(pred_target, dim=1))
        loss_adv += bce_loss(D_out, torch.FloatTensor(D_out.data.size()).fill_(source_label).to(device))
        loss_adv = loss_adv*0.01
        loss_adv.backward()
        
        optimizer.step()

        # train D
        # bring back requires_grad
        for param in model_D.parameters():
            param.requires_grad = True

        # train with source
        loss_D_source = 0
        D_out_source = model_D[0](feat_source.detach())
        loss_D_source += bce_loss(D_out_source, torch.FloatTensor(D_out_source.data.size()).fill_(source_label).to(device))
        D_out_source = model_D[1](F.softmax(pred_source.detach(),dim=1))
        loss_D_source += bce_loss(D_out_source, torch.FloatTensor(D_out_source.data.size()).fill_(source_label).to(device))
        loss_D_source.backward()

        # train with target
        loss_D_target = 0
        D_out_target = model_D[0](feat_target.detach())
        loss_D_target += bce_loss(D_out_target, torch.FloatTensor(D_out_target.data.size()).fill_(target_label).to(device))
        D_out_target = model_D[1](F.softmax(pred_target.detach(),dim=1))
        loss_D_target += bce_loss(D_out_target, torch.FloatTensor(D_out_target.data.size()).fill_(target_label).to(device))
        loss_D_target.backward()
        
        optimizer_D.step()

        if i_iter % 10 == 0:
            print('iter = {0:8d}/{1:8d}, loss_seg = {2:.3f} loss_adv = {3:.3f} loss_D_s = {4:.3f}, loss_D_t = {5:.3f}'.format(
            i_iter, args.num_steps, loss_seg.item(), loss_adv.item(), loss_D_source.item(), loss_D_target.item()))

        if i_iter >= args.num_steps_stop - 1:
            print('save model ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(args.num_steps_stop) + '.pth'))
            torch.save(model_D.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(args.num_steps_stop) + '_D.pth'))
            break

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            print('taking snapshot ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '.pth'))
            torch.save(model_D.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '_D.pth'))


if __name__ == '__main__':
    main()
