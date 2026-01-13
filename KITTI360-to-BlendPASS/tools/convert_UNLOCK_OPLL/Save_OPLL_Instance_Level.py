import os.path as osp
import glob
import numpy as np
import sys
sys.path.append('../../')
from amodalscripts.helpers.labels import id2label, labels, labelid2name
import torch
import torch.nn.functional as F
import os
import logging
import numpy as np
from tqdm import tqdm
import errno
from collections import OrderedDict
from PIL import Image
import shutil
import colorlog
def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def _getvocpallete(num_cls):
    n = num_cls
    pallete = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        pallete[j * 3 + 0] = 0
        pallete[j * 3 + 1] = 0
        pallete[j * 3 + 2] = 0
        i = 0
        while (lab > 0):
            pallete[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            pallete[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            pallete[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i = i + 1
            lab >>= 3
    return pallete
def get_color_pallete(npimg, dataset='voc'):
    out_img = Image.fromarray(npimg.astype('uint8')).convert('P')
    if dataset == 'city':
        cityspallete = [
            128, 64, 128,
            244, 35, 232,
            70, 70, 70,
            102, 102, 156,
            190, 153, 153,
            153, 153, 153,
            250, 170, 30,
            220, 220, 0,
            107, 142, 35,
            152, 251, 152,
            0, 130, 180,
            220, 20, 60,
            255, 0, 0,
            0, 0, 142,
            0, 0, 70,
            0, 60, 100,
            0, 80, 100,
            0, 0, 230,
            119, 11, 32,
        ]
        out_img.putpalette(cityspallete)
    else:
        vocpallete = _getvocpallete(256)
        out_img.putpalette(vocpallete)
    return out_img

class PseudoLabel:
    def __init__(self, inspred_list, out_dir):
        gt_instance = np.load(inspred_list[0], allow_pickle=True)
        self.prob_tar = np.zeros([1, 1])
        self.label_tar = np.zeros([1, 1])
        self.thres = []
        self.thres_percent = []
        self.number_class = 7
        self.out_dir = out_dir
        self.iter = 0
        # self.ins_thread = ins_thread

    def save_results(self):
        np.save(os.path.join(self.out_dir, 'AIns_thres_const.npy'), self.thres)
        logger.info("save done.")

    def update_pseudo_label(self, input):
        # input = F.softmax(input.detach(), dim=1)
        for i in input:
            score = i['score']
            label = i['pred_class']-11
            # if score < self.ins_thread:
            #     continue
            # logger.info(np.array(score))
            if self.iter==0:
                self.prob_tar = np.array(score).reshape(1,1)
                self.label_tar = np.array(label).reshape(1,1)
            else:
                self.prob_tar = np.append(self.prob_tar, np.array(score).reshape(1,1), axis=0)
                self.label_tar = np.append(self.label_tar, np.array(label).reshape(1,1), axis=0)
            self.iter += 1

    def get_threshold_const(self, threshold, percent=0.5):
        num_percent = []
        num_threshold = []
        for i in range(self.number_class):
            x = self.prob_tar[self.label_tar == i]
            if len(x) == 0:
                self.thres.append(0)
                continue
            x = np.sort(x)
            # self.thres_percent.append(x[np.int_(np.round(len(x) * (1-percent)))])
            self.thres_percent.append(x[np.int_(np.round(len(x) * (1-percent)))])
            logger.info('#'*10+'class is {}'.format(labelid2name[i+11])+ '#'*10)
            logger.info('The numbers of predictions, {}'.format(len(x)))
            logger.info('Under threshold, {}'.format(len(x[x>=threshold])))
            logger.info('Under percent, {}'.format(len(x[x>=x[np.int_(np.round(len(x) * (1-percent)))]])))
            num_percent.append(len(x[x>=x[np.int_(np.round(len(x) * (1-percent)))]]))
            num_threshold.append(len(x[x>=threshold]))
            if len(x[x>=threshold]) < (len(x[x>=x[np.int_(np.round(len(x) * (1-percent)))]])):
                threshold_i = x[np.int_(np.round(len(x) * (1-percent)))]
                logger.warning('The threshold has changed: {} with front {}%'.format(threshold_i, (1-percent)*100))
                logger.info('Under the new threshold, {}'.format(len(x[x >= threshold_i])))
            else:
                threshold_i = threshold
            self.thres.append(threshold_i)
        logger.info('The numbers of predictions under the threshold is {}'.format(num_threshold))
        logger.info('The numbers of predictions under the percent is {}'.format(num_percent))
        return np.array(self.thres), np.array(self.thres_percent)

def get_threshold(inspred_list, out_dir, ins_thread, threashold_percent):
    logger.info("Start inference on target dataset and get threshold of each class")

    cpseudo_label = PseudoLabel(inspred_list, out_dir)
    i =0
    for pred_path in tqdm(inspred_list):
        output = np.load(pred_path, allow_pickle=True)
        cpseudo_label.update_pseudo_label(output)
        # i += 1
        # if i>100:
        #     break
    # thres_const = cpseudo_label.get_threshold_const(thred=0.9, percent=threashold_percent)
    thres_const = cpseudo_label.get_threshold_const(threshold=ins_thread, percent=threashold_percent)
    # cpseudo_label.save_results()

    return thres_const

Path_root = '../../unmaskformer_experiments/exp-00006/Sourceonly_UNLOCK/oass_eval/pred_numpys'  ## Update with your pseudo-label generation path
Out_dir = './Save'
AIns_path = osp.join(Path_root, 'amodal_instance')
Ins_path = osp.join(Path_root, 'instance')
Num_class = 7

I_Threashold_percent = 0.3
AI_Threashold_percent = 0.5
Ins_thread = 0.5
AIns_thread = 0.3

output_folder = os.path.join(Path_root[:-(len(Path_root.split('/')[-1]))], "OPLL")
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
logger = logging.getLogger('SoftLabels')
logger.setLevel(logging.INFO)
# formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
formatter = logging.Formatter("%(asctime)s: %(message)s")
log_colors_config = {'DEBUG': 'white', 'INFO': 'cyan', 'WARNING': 'yellow', 'ERROR': 'red',
                     'CRITICAL': 'bold_red'}
formatter_console = colorlog.ColoredFormatter(fmt='%(log_color)s%(asctime)s: %(message)s', log_colors=log_colors_config)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter_console)
logger.addHandler(stream_handler)

logger_output_name = os.path.join(output_folder,
                                  'log_IoT{}wsem_AIoT{}wsem.log'.format(str(Ins_thread).replace('.', ''),
                                                                            str(AIns_thread).replace('.', '')))
if osp.exists(logger_output_name):
    logger.warning('The output file {} is exits, it will be removed.'.format(logger_output_name))
    os.remove(logger_output_name)
file_handler = logging.FileHandler(logger_output_name)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

##########
### generating pseudo instance labels
logger.info("Start generating pseudo instance labels")
Inspred_list = glob.glob(osp.join(Ins_path, '*', '*.npy'))
Inspred_list.sort()

### calculate instance threshold
I_thres_const, I_thres_percent = get_threshold(Inspred_list, Out_dir, Ins_thread, I_Threashold_percent)
logger.info('The pseudo instance hard threashold is {}'.format(I_thres_const))
logger.info('The pseudo instance percent threashold is {}'.format(I_thres_percent))

### save instance labels
I_output_folder = os.path.join(output_folder,  'instance')
if osp.exists(I_output_folder):
    logger.warning('The directory {} exist.'.format(I_output_folder))
    logger.warning('This directory will be removed')
    shutil.rmtree(I_output_folder)
mkdir(I_output_folder)
logger.info("The instance labels were saved in {}".format(I_output_folder))
for pred_path in tqdm(Inspred_list):
    output = np.load(pred_path, allow_pickle=True)
    sem_output = np.load(pred_path.replace('instance', 'semantic'), allow_pickle=True)
    sem_output = sem_output.argmax(0)
    sem_mask = np.zeros_like(sem_output, dtype=np.uint8)
    for thing_i in range(Num_class):
        sem_mask[sem_output == (thing_i+11)] = 255
    stat_instance = {}
    if len(output):
        soft_sum = np.zeros_like(output[0]['pred_mask'])
        hard_sum = np.zeros_like(output[0]['pred_mask'])
    for i in range(7):
        stat_instance[i] = 0
    for ot in output:
        score = ot['score']
        label = ot['pred_class'] - 11
        mask = ot['pred_mask']
        mkdir(os.path.join(I_output_folder, pred_path.split('/')[-2]))
        # if score < I_thres_const[label] and score < I_thres_percent[label]:
        #     continue
        # if score < I_thres_percent[label]:
        if score < I_thres_const[label]:
            soft_sum[mask==1] = 255
            continue
        mask = mask & (sem_mask)
        if np.sum(mask) == 0:
            continue
        hard_sum[mask == 1] = 255
        mask = Image.fromarray(mask)
        mask_filename = pred_path.split('/')[-2] + '/' + pred_path.split('/')[-1].replace('.npy',
                                                                                          '_cls{}_id{}.png'.format(
                                                                                              label,
                                                                                              stat_instance[label]))
        mask.save(os.path.join(I_output_folder, mask_filename))
        stat_instance[label] += 1
    soft_sum = soft_sum & (~hard_sum)
    soft_sum = soft_sum & (sem_mask)
    soft_mask = Image.fromarray(soft_sum)
    soft_mask_filename = pred_path.split('/')[-2] + '/' + pred_path.split('/')[-1].replace('.npy',
                                                                                      '_uncertain.png')
    soft_mask.save(os.path.join(I_output_folder, soft_mask_filename))
    # hard_mask = Image.fromarray(sem_mask)
    # hard_mask_filename = pred_path.split('/')[-2] + '/' + pred_path.split('/')[-1].replace('.npy',
    #                                                                                   'certain.png')
    # hard_mask.save(os.path.join(I_output_folder, hard_mask_filename))

##########
### generating pseudo amodal instance labels
logger.info("Start generating pseudo amodal instance labels")
AInspred_list = glob.glob(osp.join(AIns_path, '*', '*.npy'))
AInspred_list.sort()

### calculate amodal instance threshold
AI_thres_const, AI_thres_percent = get_threshold(AInspred_list, Out_dir, AIns_thread, AI_Threashold_percent)
logger.info('The pseudo instance hard threashold is {}'.format(AI_thres_const))
logger.info('The pseudo instance percent threashold is {}'.format(AI_thres_percent))

### save amodal instance labels
AI_output_folder = os.path.join(output_folder,  'amodal_instance')
if osp.exists(AI_output_folder):
    logger.warning('The directory {} exist.'.format(AI_output_folder))
    logger.warning('This directory will be removed')
    shutil.rmtree(AI_output_folder)
mkdir(AI_output_folder)
logger.info("The instance labels were saved in {}".format(AI_output_folder))
for pred_path in tqdm(AInspred_list):
    output = np.load(pred_path, allow_pickle=True)

    sem_output = np.load(pred_path.replace('amodal_instance', 'semantic'), allow_pickle=True)
    sem_output = sem_output.argmax(0)
    sem_mask = np.zeros_like(sem_output, dtype=np.uint8)
    for thing_i in range(Num_class):
        sem_mask[sem_output == (thing_i + 11)] = 255

    stat_ainstance = {}
    if len(output):
        soft_sum = np.zeros_like(output[0]['pred_mask'])
        hard_sum = np.zeros_like(output[0]['pred_mask'])
    for i in range(7):
        stat_ainstance[i] = 0
    for ot in output:
        score = ot['score']
        label = ot['pred_class'] - 11
        mask = ot['pred_mask']
        mkdir(os.path.join(AI_output_folder, pred_path.split('/')[-2]))
        # if score < AI_thres_const[label] and score < AI_thres_percent[label]:
        #     continue
        # if score < AI_thres_percent[label]:
        if score < AI_thres_const[label]:
            soft_sum[mask == 1] = 255
            continue
        mask = mask & (sem_mask)
        if np.sum(mask) == 0:
            continue
        hard_sum[mask == 1] = 255
        mask = Image.fromarray(mask)
        mask_filename = pred_path.split('/')[-2] + '/' + pred_path.split('/')[-1].replace('.npy',
                                                                                          '_cls{}_id{}.png'.format(
                                                                                              label,
                                                                                              stat_ainstance[label]))
        mkdir(os.path.join(AI_output_folder, pred_path.split('/')[-2]))
        mask.save(os.path.join(AI_output_folder, mask_filename))
        stat_ainstance[label] += 1
    soft_sum = soft_sum & (~hard_sum)
    soft_sum = soft_sum & (sem_mask)
    soft_mask = Image.fromarray(soft_sum)
    soft_mask_filename = pred_path.split('/')[-2] + '/' + pred_path.split('/')[-1].replace('.npy', '_uncertain.png')
    soft_mask.save(os.path.join(AI_output_folder, soft_mask_filename))
