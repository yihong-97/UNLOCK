#!/bin/bash

cd /path/to/the/unmaskformer
source ~/venv/unmaskformer/bin/activate
PYTHONPATH="</path/to/the/unmaskformer>:$PYTHONPATH" && export PYTHONPATH
python mapillaryscripts/save_panoptic_gt_labels_for_mapillary_as_pickle_files_19cls.py