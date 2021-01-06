import argparse
import time
import util
import os
import os.path as osp
import timeit
import torch
from torch.utils import data
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import math
from PIL import Image
import numpy as np
import shutil
import random

from model.deeplab_multi import DeeplabMulti
from model.discriminator import FCDiscriminator
from model.discriminator import OutspaceDiscriminator
from dataset.cbst_dataset import SrcSTDataSet, TgtSTDataSet, TestDataSet

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32) # BGR

BATCH_SIZE = 1
IGNORE_LABEL = 255
LEARNING_RATE = 2e-4
LEARNING_RATE_D = 1e-4
MOMENTUM = 0.9
NUM_CLASSES = 19
NUM_STEPS = 100000
NUM_STEPS_STOP = 20000  # early stopping
POWER = 0.9
RESTORE_FROM = './round0.pth'
RESTORE_FROM_D = './round0_D.pth'
SAVE_PRED_EVERY = 1000
SNAPSHOT_DIR = './snapshots/GTA2Cityscapes'
WEIGHT_DECAY = 0.0005
INIT_TGT_PORT = 0.5

SAVE_PATH = 'debug'
LOG_FILE = 'self_training_log'

SOURCE = 'GTA5'
INPUT_SIZE = '1280,720'
DATA_SRC_DIRECTORY = './data/GTA5'
DATA_SRC_LIST_PATH = './dataset/gta5_list/train.txt'


TARGET = 'cityscapes'
INPUT_SIZE_TARGET = '1024, 512'
DATA_TGT_DIRECTORY = './data/Cityscapes'
DATA_TGT_TRAIN_LIST_PATH = './dataset/cityscapes_list/train.txt'
DATA_TGT_TEST_LIST_PATH = './dataset/cityscapes_list/val.txt'

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-VGG Network")
    parser.add_argument("--data-src-dir", type=str, default=DATA_SRC_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-src-list", type=str, default=DATA_SRC_LIST_PATH,
                        help="Path to the file listing the images&labels in the source dataset.")
    parser.add_argument("--data-tgt-dir", type=str, default=DATA_TGT_DIRECTORY,
                        help="Path to the directory containing the target dataset.")
    parser.add_argument("--data-tgt-train-list", type=str, default=DATA_TGT_TRAIN_LIST_PATH,
                        help="Path to the file listing the images*GT labels in the target train dataset.")
    parser.add_argument("--data-tgt-test-list", type=str, default=DATA_TGT_TEST_LIST_PATH,
                        help="Path to the file listing the images*GT labels in the target test dataset.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--restore-from-D", type=str, default=RESTORE_FROM_D,
                        help="Where restore model parameters from.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--input-size-target", type=str, default=INPUT_SIZE_TARGET,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--save", type=str, default=SAVE_PATH,
                        help="Path to save result for self-training.")
    parser.add_argument('--init-tgt-port', default=INIT_TGT_PORT, type=float, dest='init_tgt_port',
                        help='The initial portion of target to determine kc')
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP,
                        help="Number of training steps for early stopping.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument('--rm-prob',
                        help='If remove the probability maps generated in every round.',
                        default=False, action='store_true')
    parser.add_argument("--log-file", type=str, default=LOG_FILE,
                        help="The name of log file.")
    parser.add_argument('--debug',help='True means logging debug info.',
                        default=False, action='store_true')
    return parser.parse_args()

args = get_arguments()

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask

def main():
    setup_seed(666)
    device = torch.device("cuda")
    save_path = args.save
    save_pseudo_label_path = osp.join(save_path, 'pseudo_label')  # in 'save_path'. Save labelIDs, not trainIDs.
    save_stats_path = osp.join(save_path, 'stats') # in 'save_path'
    save_lst_path = osp.join(save_path, 'list')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(save_pseudo_label_path):
        os.makedirs(save_pseudo_label_path)
    if not os.path.exists(save_stats_path):
        os.makedirs(save_stats_path)
    if not os.path.exists(save_lst_path):
        os.makedirs(save_lst_path)
        
    cudnn.enabled = True
    cudnn.benchmark = True
    
    logger = util.set_logger(args.save, args.log_file, args.debug)
    logger.info('start with arguments %s', args)
    
    model = DeeplabMulti(num_classes=args.num_classes)
    saved_state_dict = torch.load(args.restore_from)
    model.load_state_dict(saved_state_dict)
    model.train()
    model.to(device)
    
    # init D
    num_class_list = [2048, 19]
    model_D = nn.ModuleList([FCDiscriminator(num_classes=num_class_list[i]).train().to(device) if i<1 else OutspaceDiscriminator(num_classes=num_class_list[i]).train().to(device) for i in range(2)])
    saved_state_dict_D = torch.load(args.restore_from_D)
    model_D.load_state_dict(saved_state_dict_D)
    
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
        
    image_src_list, _, src_num = parse_split_list(args.data_src_list)
    image_tgt_list, image_name_tgt_list, tgt_num = parse_split_list(args.data_tgt_train_list)
    # portions
    tgt_portion = args.init_tgt_port

    # training crop size
    w, h = map(int, args.input_size.split(','))
    input_size = (w, h)

    w, h = map(int, args.input_size_target.split(','))
    input_size_target = (w, h)
    
    bce_loss1 = torch.nn.MSELoss()
    bce_loss2 = torch.nn.MSELoss(reduce=False, reduction='none')
    seg_loss = torch.nn.CrossEntropyLoss(ignore_index=255)
    round_idx = 3
    save_round_eval_path = osp.join(args.save,str(round_idx))
    save_pseudo_label_color_path = osp.join(save_round_eval_path, 'pseudo_label_color')
    if not os.path.exists(save_round_eval_path):
        os.makedirs(save_round_eval_path)
    if not os.path.exists(save_pseudo_label_color_path):
        os.makedirs(save_pseudo_label_color_path)
    ########## pseudo-label generation
    # evaluation & save confidence vectors
    test(model, model_D, device, save_round_eval_path, round_idx, 500, args, logger)
    conf_dict, pred_cls_num, save_prob_path, save_pred_path = val(model, model_D, device, save_round_eval_path, round_idx, tgt_num, args, logger)
    # class-balanced thresholds
    cls_thresh = kc_parameters(conf_dict, pred_cls_num, tgt_portion, round_idx, save_stats_path, args, logger)
    # pseudo-label maps generation
    label_selection(cls_thresh, tgt_num, image_name_tgt_list, round_idx, save_prob_path, save_pred_path, save_pseudo_label_path, save_pseudo_label_color_path, save_round_eval_path, args, logger)
    src_train_lst, tgt_train_lst, src_num_sel = savelst_SrcTgt(image_tgt_list, image_name_tgt_list, image_src_list, save_lst_path, save_pseudo_label_path, src_num, tgt_num, args)
    ########### model retraining
    # dataset
    srctrainset = SrcSTDataSet(args.data_src_dir, src_train_lst, max_iters=args.num_steps * args.batch_size,
                               crop_size=input_size, scale=False, mirror=False, mean=IMG_MEAN)
    tgttrainset = TgtSTDataSet(args.data_tgt_dir, tgt_train_lst, pseudo_root=save_pseudo_label_path, max_iters=args.num_steps * args.batch_size,
                               crop_size=input_size_target, scale=False, mirror=False, mean=IMG_MEAN, set='train')
    trainloader = torch.utils.data.DataLoader(srctrainset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    trainloader_iter = enumerate(trainloader)
    targetloader = torch.utils.data.DataLoader(tgttrainset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    targetloader_iter = enumerate(targetloader)

    optimizer = optim.SGD(model.parameters(),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()
    optimizer_D = optim.Adam(model_D.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D.zero_grad()
    logger.info('###### Start model retraining dataset in round {}! ######'.format(round_idx))
    
    start = timeit.default_timer()
    # start training
    interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear', align_corners=True)
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear', align_corners=True)
    
    # labels for adversarial training
    source_label = 0
    target_label = 1
    
    for i_iter in range(args.num_steps):

        lamb = 1
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
        images, labels, _, _ = batch
        images = images.to(device)
        labels = labels.long().to(device)

        feat_target, pred_target = model(images, model_D, 'target')
        pred_target = interp_target(pred_target)
        # atten_target = F.interpolate(atten_target, size=(16, 32), mode='bilinear', align_corners=True)
        
        loss_seg_tgt = seg_loss(pred_target, labels)*lamb

        D_out1 = model_D[0](feat_target)
        loss_adv1 = bce_loss1(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(source_label).to(device))
        D_out2 = model_D[1](F.softmax(pred_target, dim=1))
        loss_adv2 = bce_loss2(D_out2, torch.FloatTensor(D_out2.data.size()).fill_(source_label).to(device))
        loss_adv = loss_adv1*0.01 + loss_adv2.mean()*0.01
        loss = loss_seg_tgt + loss_adv
        loss.backward()
        
        optimizer.step()

        # train D
        # bring back requires_grad
        for param in model_D.parameters():
            param.requires_grad = True

        # train with source
        D_out_source1 = model_D[0](feat_source.detach())
        loss_D_source1 = bce_loss1(D_out_source1, torch.FloatTensor(D_out_source1.data.size()).fill_(source_label).to(device))
        D_out_source2 = model_D[1](F.softmax(pred_source.detach(),dim=1))
        loss_D_source2 = bce_loss1(D_out_source2, torch.FloatTensor(D_out_source2.data.size()).fill_(source_label).to(device))
        loss_D_source = loss_D_source1 + loss_D_source2
        loss_D_source.backward()

        # train with target
        D_out_target1 = model_D[0](feat_target.detach())
        loss_D_target1 = bce_loss1(D_out_target1, torch.FloatTensor(D_out_target1.data.size()).fill_(target_label).to(device))
        D_out_target2 = model_D[1](F.softmax(pred_target.detach(),dim=1))
        weight_target = bce_loss2(D_out_target2, torch.FloatTensor(D_out_target2.data.size()).fill_(target_label).to(device))
        loss_D_target2 = weight_target.mean()
        loss_D_target = loss_D_target1 + loss_D_target2
        loss_D_target.backward()
       
        optimizer_D.step()

        if i_iter % 10 == 0:
            print('iter={0:8d}/{1:8d}, seg={2:.3f} seg_tgt={3:.3f} adv={4:.3f} adv1={5:.3f} adv2={6:.3f} src1={7:.3f} src2={8:.3f} tgt1={9:.3f} tgt2={10:.3f} D1={11:.3f} D2={12:.3f}'.format(
            i_iter, args.num_steps, loss_seg.item(), loss_seg_tgt.item(), loss_adv.item(), loss_adv1.item(), loss_adv2.mean().item(), loss_D_source1.item(), loss_D_source2.item(), 
            loss_D_target1.item(), loss_D_target2.item(), loss_D_source.item(), loss_D_target.item()))

        if i_iter >= args.num_steps_stop - 1:
            print('save model ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(args.num_steps_stop) + '.pth'))
            torch.save(model_D.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(args.num_steps_stop) + '_D.pth'))
            break

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            print('taking snapshot ...')
            test(model, model_D, device, save_round_eval_path, round_idx, 500, args, logger)
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '.pth'))
            torch.save(model_D.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '_D.pth'))

    
    end = timeit.default_timer()
    logger.info('###### Finish model retraining dataset in round {}! Time cost: {:.2f} seconds. ######'.format(round_idx, end - start))
    # test self-trained model in target domain test set
    test(model, model_D, device, save_round_eval_path, round_idx, 500, args, logger)

def savelst_SrcTgt(image_tgt_list, image_name_tgt_list, image_src_list, save_lst_path, save_pseudo_label_path, src_num, tgt_num, args):
    src_train_lst = osp.join(save_lst_path,'src_train.txt')
    tgt_train_lst = osp.join(save_lst_path, 'tgt_train.txt')

    # generate src train list
    with open(src_train_lst, 'w') as f:
        for idx in range(src_num):
            f.write("%s\n" % (image_src_list[idx]))
    # generate tgt train list
    with open(tgt_train_lst, 'w') as f:
        for idx in range(tgt_num):
            image_tgt_path = osp.join(save_pseudo_label_path,image_name_tgt_list[idx])
            f.write("%s\t%s\n" % (image_tgt_list[idx], image_tgt_path))

    return src_train_lst, tgt_train_lst, src_num
          
def label_selection(cls_thresh, tgt_num, image_name_tgt_list, round_idx, save_prob_path, save_pred_path, save_pseudo_label_path, save_pseudo_label_color_path, save_round_eval_path, args, logger):
    logger.info('###### Start pseudo-label generation in round {} ! ######'.format(round_idx))
    start_pl = time.time()
    for idx in range(tgt_num):
        sample_name = image_name_tgt_list[idx].split('.')[0]
        probmap_path = osp.join(save_prob_path, '{}.npy'.format(sample_name))
        pred_path = osp.join(save_pred_path, '{}.png'.format(sample_name))
        pred_prob = np.load(probmap_path)
        pred_label_trainIDs = np.asarray(Image.open(pred_path))
        save_wpred_vis_path = osp.join(save_round_eval_path, 'weighted_pred_vis')
        if not os.path.exists(save_wpred_vis_path):
            os.makedirs(save_wpred_vis_path)
        weighted_prob = pred_prob/cls_thresh
        weighted_pred_trainIDs = np.asarray(np.argmax(weighted_prob, axis=2), dtype=np.uint8)
        # save weighted predication
        wpred_label_col = weighted_pred_trainIDs.copy()
        wpred_label_col = colorize_mask(wpred_label_col)
        wpred_label_col.save('%s/%s_color.png' % (save_wpred_vis_path, sample_name))
        weighted_conf = np.amax(weighted_prob, axis=2)
        pred_label_trainIDs = weighted_pred_trainIDs.copy()
        pred_label_trainIDs[weighted_conf < 1] = 255 # '255' in cityscapes indicates 'unlabaled' for trainIDs

        # pseudo-labels with labelID
        pseudo_label_trainIDs = pred_label_trainIDs.copy()
        # save colored pseudo-label map
        pseudo_label_col = colorize_mask(pseudo_label_trainIDs)
        pseudo_label_col.save('%s/%s_color.png' % (save_pseudo_label_color_path, sample_name))
        # save pseudo-label map with label IDs
        pseudo_label_save = Image.fromarray(pseudo_label_trainIDs.astype(np.uint8))
        pseudo_label_save.save('%s/%s.png' % (save_pseudo_label_path, sample_name))

    # remove probability maps
    if args.rm_prob:
        shutil.rmtree(save_prob_path)

    logger.info('###### Finish pseudo-label generation in round {}! Time cost: {:.2f} seconds. ######'.format(round_idx,time.time() - start_pl))

def parse_split_list(list_name):
    image_list = []
    image_name_list = []
    file_num = 0
    with open(list_name) as f:
        for item in f.readlines():
            fields = item.strip()
            image_name = fields.split('/')[-1]
            image_list.append(fields)
            image_name_list.append(image_name)
            file_num += 1
    return image_list, image_name_list, file_num
       
def kc_parameters(conf_dict, pred_cls_num, tgt_portion, round_idx, save_stats_path, args, logger):
    logger.info('###### Start kc generation in round {} ! ######'.format(round_idx))
    start_kc = time.time()
    # threshold for each class
    cls_thresh = np.ones(args.num_classes,dtype = np.float32)
    cls_sel_size = np.zeros(args.num_classes, dtype=np.float32)
    cls_size = np.zeros(args.num_classes, dtype=np.float32)
    for idx_cls in np.arange(0, args.num_classes):
        cls_size[idx_cls] = pred_cls_num[idx_cls]
        if conf_dict[idx_cls] != None:
            conf_dict[idx_cls].sort(reverse=True) # sort in descending order
            len_cls = len(conf_dict[idx_cls])
            cls_sel_size[idx_cls] = int(math.floor(len_cls * tgt_portion))
            len_cls_thresh = int(cls_sel_size[idx_cls])
            if len_cls_thresh != 0:
            	if conf_dict[idx_cls][len_cls_thresh-1]<0.9:
            		cls_thresh[idx_cls] = conf_dict[idx_cls][len_cls_thresh-1]
            	else:
            		cls_thresh[idx_cls] = 0.9
            conf_dict[idx_cls] = None
    # save thresholds
    np.save(save_stats_path + '/cls_thresh_round' + str(round_idx) + '.npy', cls_thresh)
    np.save(save_stats_path + '/cls_sel_size_round' + str(round_idx) + '.npy', cls_sel_size)
    logger.info('###### Finish kc generation in round {}! Time cost: {:.2f} seconds. ######'.format(round_idx,time.time() - start_kc))
    return cls_thresh

def val(model, model_D, device, save_round_eval_path, round_idx, tgt_num, args, logger):
    """Create the model and start the evaluation process."""
    ## scorer
    scorer = ScoreUpdater(args.num_classes, tgt_num, logger)
    scorer.reset()

    ## test data loader
    testloader = data.DataLoader(TestDataSet(args.data_tgt_dir, args.data_tgt_train_list, crop_size=(1024,512), mean=IMG_MEAN, scale=False, mirror=False, set='train'),
                                        batch_size=1, shuffle=False, pin_memory=True)
    model.eval()
    model_D.eval()
    model.to(device)
    model_D.to(device)

    ## upsampling layer
    interp = nn.Upsample(size=(512, 1024), mode='bilinear', align_corners=True)

    ## output of deeplab is logits, not probability
    softmax2d = nn.Softmax2d()

    ## output folder
    save_pred_vis_path = osp.join(save_round_eval_path, 'pred_vis')
    save_prob_path = osp.join(save_round_eval_path, 'prob')
    save_pred_path = osp.join(save_round_eval_path, 'pred')
    if not os.path.exists(save_pred_vis_path):
        os.makedirs(save_pred_vis_path)
    if not os.path.exists(save_prob_path):
        os.makedirs(save_prob_path)
    if not os.path.exists(save_pred_path):
        os.makedirs(save_pred_path)

    # saving output data
    conf_dict = {k: [] for k in range(args.num_classes)}
    pred_cls_num = np.zeros(args.num_classes)
    ## evaluation process
    logger.info('###### Start evaluating target domain train set in round {}! ######'.format(round_idx))
    start_eval = time.time()
    with torch.no_grad():
        for index, batch in enumerate(testloader):
            image, label, _, name = batch
            label = label.cpu().data[0].numpy()
            _, pred = model(image.to(device), model_D, 'target')
            output = softmax2d(interp(pred)).cpu().data[0].numpy()
            output = output.transpose(1,2,0)
            amax_output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
            conf = np.amax(output,axis=2)
            # score
            pred_label = amax_output.copy()
            scorer.update(pred_label.flatten(), label.flatten(), index)

            # save visualized seg maps & predication prob map
            amax_output_col = colorize_mask(amax_output)
            name = name[0].split('/')[-1]
            image_name = name.split('.')[0]
            # prob
            np.save('%s/%s.npy' % (save_prob_path, image_name), output)
            # trainIDs/vis seg maps
            amax_output = Image.fromarray(amax_output)
            amax_output.save('%s/%s.png' % (save_pred_path, image_name))
            amax_output_col.save('%s/%s_color.png' % (save_pred_vis_path, image_name))

            # save class-wise confidence maps
            for idx_cls in range(args.num_classes):
                idx_temp = pred_label == idx_cls
                pred_cls_num[idx_cls] = pred_cls_num[idx_cls] + np.sum(idx_temp)
                if idx_temp.any():
                    conf_cls_temp = conf[idx_temp].astype(np.float32)
                    len_cls_temp = conf_cls_temp.size
                    # downsampling by ds_rate
                    conf_cls = conf_cls_temp[0:len_cls_temp:4]
                    conf_dict[idx_cls].extend(conf_cls)
    logger.info('###### Finish evaluating target domain train set in round {}! Time cost: {:.2f} seconds. ######'.format(round_idx, time.time()-start_eval))

    return conf_dict, pred_cls_num, save_prob_path, save_pred_path  # return the dictionary containing all the class-wise confidence vectors

def test(model, model_D, device, save_round_eval_path, round_idx, test_num, args, logger):
    """Create the model and start the evaluation process."""
    ## scorer
    scorer = ScoreUpdater(args.num_classes, test_num, logger)
    scorer.reset()

    ## test data loader
    testloader = data.DataLoader(TestDataSet(args.data_tgt_dir, args.data_tgt_test_list, crop_size=(1024,512), mean=IMG_MEAN, scale=False, mirror=False, set='val'),
                                        batch_size=1, shuffle=False, pin_memory=True)
    model.eval()
    model_D.eval()
    model.to(device)
    model_D.to(device)

    ## upsampling layer
    interp = nn.Upsample(size=(1024, 2048), mode='bilinear', align_corners=True)
    
    save_test_vis_path = osp.join(save_round_eval_path, 'testSet_vis')
    if not os.path.exists(save_test_vis_path):
        os.makedirs(save_test_vis_path)

    ## evaluation process
    logger.info('###### Start evaluating in target domain test set in round {}! ######'.format(round_idx))
    start_eval = time.time()
    with torch.no_grad():
        for index, batch in enumerate(testloader):
            image, label, _, name = batch
            label = label.cpu().data[0].numpy()
            _, output = model(image.to(device), model_D, 'target')
            output = interp(output).cpu().data[0].numpy()
            output = output.transpose(1,2,0)
            amax_output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
            pred_label = amax_output.copy()
            scorer.update(pred_label.flatten(), label.flatten(), index)
            # save visualized seg maps & predication prob map
            amax_output_col = colorize_mask(amax_output)
            name = name[0].split('/')[-1]
            image_name = name.split('.')[0]
            # vis seg maps
            amax_output_col.save('%s/%s_color.png' % (save_test_vis_path, image_name))

    logger.info('###### Finish evaluating in target domain test set in round {}! Time cost: {:.2f} seconds. ######'.format(round_idx, time.time()-start_eval))

class ScoreUpdater(object):
    # only IoU are computed. accu, cls_accu, etc are ignored.
    def __init__(self, c_num, x_num, logger=None, label=None, info=None):
        self._confs = np.zeros((c_num, c_num))
        self._per_cls_iou = np.zeros(c_num)
        self._logger = logger
        self._label = label
        self._info = info
        self._num_class = c_num
        self._num_sample = x_num

    @property
    def info(self):
        return self._info

    def reset(self):
        self._start = time.time()
        self._computed = np.zeros(self._num_sample) # one-dimension
        self._confs[:] = 0

    def fast_hist(self,label, pred_label, n):
        k = (label >= 0) & (label < n)
        return np.bincount(n * label[k].astype(int) + pred_label[k], minlength=n ** 2).reshape(n, n)

    def per_class_iu(self,hist):
        return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

    def do_updates(self, conf, i, computed=True):
        if computed:
            self._computed[i] = 1
        self._per_cls_iou = self.per_class_iu(conf)

    def update(self, pred_label, label, i, computed=True):
        conf = self.fast_hist(label, pred_label, self._num_class)
        self._confs += conf
        self.do_updates(self._confs, i, computed)
        self.scores(i)

    def scores(self, i=None, logger=None):
        x_num = self._num_sample
        ious = np.nan_to_num( self._per_cls_iou )

        logger = self._logger if logger is None else logger
        if logger is not None:
            if i is not None:
                speed = 1. * self._computed.sum() / (time.time() - self._start)
                logger.info('Done {}/{} with speed: {:.2f}/s'.format(i + 1, x_num, speed))
            name = '' if self._label is None else '{}, '.format(self._label)
            logger.info('{}mean iou: {:.2f}%'. \
                        format(name, np.mean(ious) * 100))
            with util.np_print_options(formatter={'float': '{:5.2f}'.format}):
                logger.info('\n{}'.format(ious * 100))

        return ious


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

if __name__ == '__main__':
    main()
