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
        self.number_class = 7
        self.out_dir = out_dir
        self.iter = 0

    def save_results(self):
        np.save(os.path.join(self.out_dir, 'AIns_thres_const.npy'), self.thres)
        print("save done.")

    def update_pseudo_label(self, input):
        # input = F.softmax(input.detach(), dim=1)
        for i in input:
            score = i['score']
            label = i['pred_class']-11
            # print(np.array(score))
            if self.iter==0:
                self.prob_tar = np.array(score).reshape(1,1)
                self.label_tar = np.array(label).reshape(1,1)
            else:
                self.prob_tar = np.append(self.prob_tar, np.array(score).reshape(1,1), axis=0)
                self.label_tar = np.append(self.label_tar, np.array(label).reshape(1,1), axis=0)
            self.iter += 1

    def get_threshold_const(self, thred, percent=0.5):
        for i in range(self.number_class):
            x = self.prob_tar[self.label_tar == i]
            if len(x) == 0:
                self.thres.append(0)
                continue
            x = np.sort(x)
            self.thres.append(x[np.int_(np.round(len(x) * (1-percent)))])
        self.thres = np.array(self.thres)
        self.thres[self.thres > thred] = thred
        return self.thres

def get_threshold(inspred_list, out_dir, thread, threashold_percent):
    logger = logging.getLogger("pseudo_label.trainer")
    logger.info("Start inference on target dataset and get threshold of each class")

    cpseudo_label = PseudoLabel(inspred_list, out_dir)
    i =0
    for pred_path in tqdm(inspred_list):
        output = np.load(pred_path, allow_pickle=True)
        cpseudo_label.update_pseudo_label(output)
        # i += 1
        # if i>10:
        #     break
    thres_const = cpseudo_label.get_threshold_const(thred=thread, percent=threashold_percent)
    # cpseudo_label.save_results()

    return thres_const


Path_root = '../../unmaskformer_experiments/exp-00006/Sourceonly_UNLOCK/oass_eval/pred_numpys'  ## Update with your pseudo-label generation path
Out_dir = './Save'
AIns_path = osp.join(Path_root, 'amodal_instance')
Ins_path = osp.join(Path_root, 'instance')
Num_class = 7

AI_Threashold_percent = 0.1
AIns_thread = 0.95

output_folder = Path_root[:-(len(Path_root.split('/')[-1]))]

print("Start generating pseudo amodal instance labels")
AInspred_list = glob.glob(osp.join(AIns_path, '*', '*.npy'))
AInspred_list.sort()

AI_thres_const = get_threshold(AInspred_list, Out_dir, AIns_thread, AI_Threashold_percent)
print('The pseudo instance threashold is {}'.format(AI_thres_const))

AI_output_folder = os.path.join(output_folder, "ADCL", 'amodal_instance')
if osp.exists(AI_output_folder):
    print('The directory {} exist.'.format(AI_output_folder))
    shutil.rmtree(AI_output_folder)
mkdir(AI_output_folder)
print("The instance labels were saved in {}".format(AI_output_folder))

for pred_path in tqdm(AInspred_list):
    output = np.load(pred_path, allow_pickle=True)
    sem_output = np.load(pred_path.replace('amodal_instance', 'semantic'), allow_pickle=True)
    sem_output = sem_output.argmax(0)
    sem_mask = np.zeros_like(sem_output, dtype=np.uint8)
    for thing_i in range(Num_class):
        sem_mask[sem_output == (thing_i + 11)] = 1

    stat_ainstance = {}
    for i in range(7):
        stat_ainstance[i] = 0
    ains_allmask = np.zeros_like(sem_output, dtype=np.uint8)
    for ot in output:
        score = ot['score']
        label = ot['pred_class'] - 11
        mask = ot['pred_mask']
        if score < AI_thres_const[label]:
            continue
        ains_allmask += mask
    ains_allmask = ains_allmask * sem_mask
    ains_allmask[ains_allmask > 1] = 255
    for ot in output:
        score = ot['score']
        label = ot['pred_class'] - 11
        mask = ot['pred_mask']
        mkdir(os.path.join(AI_output_folder, pred_path.split('/')[-2]))
        if score < AI_thres_const[label]:
            continue
        mask = mask & (sem_mask)
        mask = ains_allmask * mask
        if not np.count_nonzero(mask):
            continue
        if np.sum(mask==255)*2 > np.count_nonzero(mask):
            continue
        mask = Image.fromarray(mask)
        mask_filename = pred_path.split('/')[-2] + '/' + pred_path.split('/')[-1].replace('.npy',
                                                                                          '_cls{}_id{}.png'.format(
                                                                                              label,
                                                                                              stat_ainstance[label]))
        mkdir(os.path.join(AI_output_folder, pred_path.split('/')[-2]))
        mask.save(os.path.join(AI_output_folder, mask_filename))
        stat_ainstance[label] += 1
