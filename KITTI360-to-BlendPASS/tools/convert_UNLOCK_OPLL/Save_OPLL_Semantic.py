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
    def __init__(self, sempred_list, out_dir):
        gt_semantic = np.load(sempred_list[0], allow_pickle=True)
        c, h, w = gt_semantic.shape
        self.prob_tar = np.zeros([1, h, w])
        self.label_tar = np.zeros([1, h, w])
        self.thres = []
        self.number_class = c
        self.out_dir = out_dir
        self.iter = 0

    def save_results(self, thred, percent):
        save_path =os.path.join(self.out_dir, 'Sem_T{}P{}_const.npy'.format(str(thred).replace('.', ''), str(int(percent*100))))
        np.save(save_path, self.thres)
        logger.info("the constant is save done in {}.".format(save_path))

    def update_pseudo_label(self, input):
        # input = F.softmax(input.detach(), dim=1)
        prob, label = torch.max(input, dim=1)
        prob_np = prob.cpu().numpy()
        label_np = label.cpu().numpy()
        # print(self.iter)
        if self.iter==0:
            self.prob_tar = prob_np
            self.label_tar = label_np
        else:
            self.prob_tar = np.append(self.prob_tar, prob_np, axis=0)
            self.label_tar = np.append(self.label_tar, label_np, axis=0)
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
        logger.info('Under percent, {}'.format(self.thres))
        self.thres[self.thres > thred] = thred
        logger.info('Under percent with threshold, {}'.format(self.thres))
        return self.thres

def get_threshold(sempred_list, out_dir, threashold_percent, threashold):

    logger.info("Start inference on target dataset and get threshold of each class")
    cpseudo_label = PseudoLabel(sempred_list, out_dir)
    # i =0
    for pred_path in tqdm(sempred_list):
        output = torch.from_numpy(np.load(pred_path, allow_pickle=True)).unsqueeze(0)
        cpseudo_label.update_pseudo_label(output)
        # i += 1
        # if i>100:
        #     break
    thres_const = cpseudo_label.get_threshold_const(thred=threashold, percent=threashold_percent)
    # cpseudo_label.save_results(thred=threashold, percent=threashold_percent)

    return thres_const


Path_root = '../../unmaskformer_experiments/exp-00006/Sourceonly_UNLOCK/oass_eval/pred_numpys'  ## Update with your pseudo-label generation path
Sem_path = osp.join(Path_root, 'semantic')
Threashold_percent = 0.8
Threashold = 0.5
Num_class = 18

Output_folder = Path_root[:-(len(Path_root.split('/')[-1]))]
Output_folder = os.path.join(Output_folder, "OPLL")

Save_folder = os.path.join(Output_folder, "semantic")
mkdir(Save_folder)
####
## log
logger = logging.getLogger('SoftSemanticLabels')
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

## log file
logger_output_name = os.path.join(Output_folder,
                                  'Sem_log_T{}P{}.log'.format(str(Threashold).replace('.', ''),
                                                                            str(int(Threashold_percent*100))))
if osp.exists(logger_output_name):
    logger.warning('The output file {} is exits, it will be removed.'.format(logger_output_name))
    os.remove(logger_output_name)
file_handler = logging.FileHandler(logger_output_name)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


sempred_list = glob.glob(osp.join(Sem_path, '*', '*.npy'))
sempred_list.sort()

thres_const = get_threshold(sempred_list, Output_folder, Threashold_percent, Threashold)

logger.info("Start semantic testing")






for index, pred_path in enumerate(sempred_list):
    if index % 100 == 0:
        logger.info("{} processed".format(index))
    output = np.load(pred_path, allow_pickle=True)
    # output = pred.max(1)[1]


    # save the pseudo label
    # pred = pred.cpu().numpy().squeeze()
    pred_max = np.max(output, 0)
    pred_label = output.argmax(0)
    for i in range(Num_class):
        pred_label[(pred_max < thres_const[i]) * (pred_label == i)] = 255
    mask = get_color_pallete(pred_label, "city")
    mask_filename = pred_path.split('/')[-2] + '/' +pred_path.split('/')[-1].replace('.npy', '.png')
    mkdir(os.path.join(Save_folder, pred_path.split('/')[-2]))
    mask.save(os.path.join(Save_folder, mask_filename))
    # np.save(os.path.join(output_folder, mask_filename.replace('.png', '.npy')), pred_label)
