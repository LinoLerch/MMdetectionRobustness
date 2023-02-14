import argparse
import mmcv
import numpy as np
import pandas as pd
import scipy.stats as stats

from mmdet.datasets import build_dataset

def parse_args():
    parser = argparse.ArgumentParser(
        description='Statisical significance test')
    parser.add_argument('tpfp1', help='path to TruePos-FalsePos List of model 1')
    parser.add_argument('tpfp2', help='path to TruePos-FalsePos List of model 2')
    parser.add_argument('tpfp3', help='path to TruePos-FalsePos List of model 3')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('names', type=str, nargs='+', help='names of the three models')
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

def print_results(confusion_mat, n1, n2):

    # convert to a pandas DataFrame and add sum at the end of columns/rows
    df = pd.DataFrame(confusion_mat, columns=[f'{n2}-C', f'{n2}-W'], index=[f'{n1}-C', f'{n1}-W'], dtype=int)
    df.loc['sum'] = df.sum()
    df['sum'] = df.sum(axis=1)

    # calculate accuracies 
    sum_imgs = df['sum']['sum']
    acc2 = round((df[f'{n2}-C']['sum'] / sum_imgs), 4)
    acc1 = round((df['sum'][f'{n1}-C'] / sum_imgs), 4)
        
    print(f'-------------------------\n {n1} vs. {n2}: \n {df} \n  Accuracy: {n1}:{acc1}, {n2}:{acc2}\n')
    mcnemar(confusion_mat)

def mcnemar(confusion_mat):
    b = confusion_mat[0,1]
    c = confusion_mat[1,0]
    chi_square = (abs(b - c) - 1)**2 / (b + c)
    p_value = 1 - stats.chi2.cdf(chi_square, 1)

    # significance level
    alpha = 0.01

    if p_value <= alpha:
        conclusion = "Null Hypothesis rejected."
    else: 
        conclusion = "Failed to reject the null hypothesis."

    print(f"McNemar's chi^2 statistic is: {round(chi_square,4)} and p value is: {round(p_value,6)}")
    print(conclusion)



def main():
    args = parse_args()
    
    cfg = mmcv.Config.fromfile(args.config)
    dataset = build_dataset(cfg.data.test)
    tpfp1 = mmcv.load(args.tpfp1)
    tpfp2 = mmcv.load(args.tpfp2)
    tpfp3 = mmcv.load(args.tpfp3)
    name1, name2, name3 = args.names

    confusion_mat1 = model_confusion_matrix(tpfp1, tpfp2, dataset)
    print_results(confusion_mat1, name1, name2)

    confusion_mat2 = model_confusion_matrix(tpfp1, tpfp3, dataset)
    print_results(confusion_mat2, name1, name3)
    
    confusion_mat3 = model_confusion_matrix(tpfp2, tpfp3, dataset)
    print_results(confusion_mat3, name2, name3)


if __name__ == '__main__':
    main()