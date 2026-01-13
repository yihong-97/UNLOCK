#!/usr/bin/python
#
# Converts the polygonal annotations of the Cityscapes dataset
# to images, where pixel values encode the ground truth classes and the
# individual instance of that classes.
#
# The Cityscapes downloads already include such images
#   a) *color.png             : the class is encoded by its color
#   b) *labelIds.png          : the class is encoded by its ID
#   c) *instanceIds.png       : the class and the instance are encoded by an instance ID
# 
# With this tool, you can generate option
#   d) *instanceTrainIds.png  : the class and the instance are encoded by an instance training ID
# This encoding might come handy for training purposes. You can use
# the file labes.py to define the training IDs that suit your needs.
# Note however, that once you submit or evaluate results, the regular
# IDs are needed.
#
# Please refer to 'json2instanceImg.py' for an explanation of instance IDs.
#
# Uses the converter tool in 'json2instanceImg.py'
# Uses the mapping defined in 'labels.py'
#

# python imports
from __future__ import print_function, absolute_import, division
import os, glob, sys

# cityscapes imports
from amodalscripts.helpers.csHelpers import printError
from amodalscripts.preparation.json2instanceImg import json2instanceImg
import cvbase as cvb

def make_json_dict(imgs, anns):
    imgs_dict = {}
    anns_dict = {}
    for ann in anns:
        image_id = ann["image_id"]
        if not image_id in anns_dict:
            anns_dict[image_id] = []
            anns_dict[image_id].append(ann)
        else:
            anns_dict[image_id].append(ann)

    for img in imgs:
        image_id = img['id']
        imgs_dict[image_id] = img['file_name']

    return imgs_dict, anns_dict
# The main method
def main():
    # Where to look for Cityscapes
    # if 'CITYSCAPES_DATASET' in os.environ:
    #     cityscapesPath = os.environ['CITYSCAPES_DATASET']
    # else:
    #     cityscapesPath = os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','..')
    # # how to search for all ground truth
    apspath = '../../data/APS_KITTI_360'
    data_split = 'val'
    # searchFine   = os.path.join( apspath, 'inmodal_add_stuff', 'val', "*.png" )
    # searchCoarse = os.path.join( cityscapesPath , "gtCoarse" , "*" , "*" , "*_gt*_polygons.json" )
    search_json =  os.path.join( apspath , 'annotations', data_split+".json" )
    # search files
    # filesFine = glob.glob( searchFine )
    # filesFine.sort()

    anns = cvb.load(search_json)
    imgs_info = anns['images']
    anns_info = anns["annotations"]
    imgs_dict, anns_dict = make_json_dict(imgs_info, anns_info)
    # filesCoarse = glob.glob( searchCoarse )
    # filesCoarse.sort()

    # concatenate fine and coarse
    # files = filesFine #+ filesCoarse
    # files = filesFine # use this line if fine is enough for now.

    # quit if we did not find anything
    # if not files:
    #     printError( "Did not find any files. Please consult the README." )

    # a bit verbose
    print("Processing {} annotation files".format(len(anns_dict)))

    # iterate through files
    progress = 0
    print("Progress: {:>3} %".format( progress * 100 / len(anns_dict) ), end=' ')
    for img_id in anns_dict.keys():
        # create the output filename
        img_name = imgs_dict[img_id]
        dst = os.path.join(apspath, 'Instance', data_split,  img_name.replace( ".png" , "_instanceTrainIds.png" ))
        img_anns = anns_dict[img_id]
        # do the conversion
        try:
            json2instanceImg(img_anns, dst , "trainIds" )
        except:
            print("Failed to convert: {}".format(img_anns))
            raise

        # status
        progress += 1
        print("\rProgress: {:>3} %".format( progress * 100 / len(anns_dict) ), end=' ')
        sys.stdout.flush()


# call the main
if __name__ == "__main__":
    main()
