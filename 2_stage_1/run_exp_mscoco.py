from __future__ import division
from __future__ import print_function

import dateutil
import dateutil.tz
import datetime
import argparse
import pprint
from shutil import copyfile
import os

from misc.dataloader import DataLoader
from stageI.model import CondGAN
from stageI.trainer_mscoco import CondGANTrainer_mscoco
from misc.utils import mkdir_p
from misc.config import cfg, cfg_from_file


def parse_args():
    parser = argparse.ArgumentParser(description='Train a GAN network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='stageI/cfg/mscoco.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=-1, type=int)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.gpu_id != -1:
        cfg.GPU_ID = args.gpu_id
    print('Using config:')
    pprint.pprint(cfg)

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

    tfrecord_path = 'Data/%s/%s.tfrecords' % \
        (cfg.DATASET_NAME, cfg.DATASET.TFRECORDS)
    crop_size = cfg.TRAIN.LR_IMSIZE
    dataset = DataLoader(tfrecord_path, [crop_size, crop_size],
        num_examples=cfg.DATASET.NUM_EXAMPLES)
    if cfg.TRAIN.FLAG:
        ckt_logs_dir = "ckt_logs/%s/%s_%s" % \
            (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)
        mkdir_p(ckt_logs_dir)
    else:
        s_tmp = cfg.TRAIN.PRETRAINED_MODEL
        ckt_logs_dir = s_tmp[:s_tmp.find('.ckpt')]

    model = CondGAN(
        image_shape=dataset.image_shape
    )

    copyfile(os.path.join('stageI', 'cfg', 'mscoco.yml'), os.path.join(ckt_logs_dir, 'mscoco.yml'))

    algo = CondGANTrainer_mscoco(
        model=model,
        dataset=dataset,
        ckt_logs_dir=ckt_logs_dir
    )
    if cfg.TRAIN.FLAG:
        algo.train()
    else:
        algo.evaluate()
