#%%
from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt
import pylab
from __future__ import division
from __future__ import print_function
pylab.rcParams['figure.figsize'] = (10.0, 8.0)
dataDir='Data/mscoco'
dataType='train2014'
annFile='%s/annotations/instances_%s.json'%(dataDir,dataType)
coco=COCO(annFile)
#%%
cats = coco.loadCats(coco.getCatIds())

supcats = set([str(cat['supercategory']) for cat in cats])

selected_supers = ['kitchen', 'indoor', 'electronic', 'furniture', 'appliance']
selected_subs = []
for cat in cats:
    if cat['supercategory'] in selected_supers:
        selected_subs.append(str(cat['name']))
print(selected_subs)

imgIds = []
for sub in selected_subs:
    catIds = coco.getCatIds(catNms=sub);
    imgIds.extend(coco.getImgIds(catIds=catIds));
imgIds = list(np.unique(imgIds))

image_names = [img['file_name'] for img in coco.loadImgs(imgIds)]