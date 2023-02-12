import argparse
import mmcv
import numpy as np

from mmdet.datasets import build_dataset

def parse_args():
    parser = argparse.ArgumentParser(
        description='Statisical significance test')
    parser.add_argument('tpfp1', help='path to TruePos-FalsePos List of model 1')
    parser.add_argument('tpfp2', help='path to TruePos-FalsePos List of model 2')
    parser.add_argument('config', help='test config file path')
    # parser.add_argument('save_dir', help='directory where results will be saved')

    args = parser.parse_args()
    return args

def model_confusion_matrix(tpfp1, tpfp2, dataset):
    """Calculate the confusion matrix comparing model 1 and 2.

    Args:
        tpfp1   (np.ndarray): TruePos-FalsePos list of model 1
        tpfp2   (np.ndarray): TruePos-FalsePos list of model 2
        dataset (mmcv.Dataset): dataset for ground truth
    """
    confusion_mat = np.zeros(shape=(2,2))

    for idx, (tp1_img, tp2_img)  in enumerate(zip(tpfp1, tpfp2)):
        # ground truth
        ann = dataset.get_ann_info(idx)
        #gt_bboxes = ann['bboxes']
        gt_labels = ann['labels']
        gt_count = len(gt_labels)
        
        tp1, fp1 = tp1_img
        im1_correct = (tp1 == gt_count and fp1 == 0)
        
        tp2, fp2 = tp2_img
        im2_correct = (tp2 == gt_count and fp2 == 0)

        if im1_correct & im2_correct:
            confusion_mat[0,0] += 1
        elif im1_correct & ~im2_correct:
            confusion_mat[0,1] += 1
        elif ~im1_correct & im2_correct:
            confusion_mat[1,0] += 1
        else:
            confusion_mat[1,1] += 1
        
    return confusion_mat

def main():
    args = parse_args()
    
    cfg = mmcv.Config.fromfile(args.config)
    dataset = build_dataset(cfg.data.test)
    tpfp1 = mmcv.load(args.tpfp1)
    tpfp2 = mmcv.load(args.tpfp2)

    confusion_mat = model_confusion_matrix(tpfp1, tpfp2, dataset)
    print(confusion_mat)


if __name__ == '__main__':
    main()