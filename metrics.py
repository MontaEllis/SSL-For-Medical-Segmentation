import torchio as tio
from pathlib import Path
import torch
import numpy as np
import copy
from torchio.transforms import (
    RandomFlip,
    RandomAffine,
    RandomElasticDeformation,
    RandomNoise,
    RandomMotion,
    RandomBiasField,
    RescaleIntensity,
    Resample,
    ToCanonical,
    ZNormalization,
    CropOrPad,
    HistogramStandardization,
    OneOf,
    Compose,
)



# predict_dir = 'G:/PHD/gan_medical/Results/binary'
# labels_dir = 'E:/chencheng/data/PCA 22/label/test'

predict_dir = ''
labels_dir = ''

# predict_dir = 'G:/PHD/Gabor v2/Results/TOF/resunet/Binary'
# predict_dir = 'G:/PHD/SSL/pi-trans/Results/SSL/our 55/binary'
# labels_dir = 'E:/chencheng/data/LungVessel/test/selected label'
# labels_dir = 'E:/chencheng/data/TOF MIDAS/test/label'

# labels_dir ='E:/chencheng/data/TOF SSL/35/test/label'




def do_subject(image_paths, label_paths):
    for (image_path, label_path) in zip(image_paths, label_paths):
        subject = tio.Subject(
            pred=tio.ScalarImage(image_path),
            gt=tio.LabelMap(label_path),
        )
        subjects.append(subject)

images_dir = Path(predict_dir)
labels_dir = Path(labels_dir)

image_paths = sorted(images_dir.glob('*.mhd'))
label_paths = sorted(labels_dir.glob('*.mhd'))\



subjects = []
do_subject(image_paths, label_paths)

training_set = tio.SubjectsDataset(subjects)


toc = ToCanonical()

acc_summary = []
pre_summary = []
rec_summary = []
dice_summary = []

for i,subj in enumerate(training_set):
    gt = subj['gt'][tio.DATA]

    # subj = toc(subj)
    pred = subj['pred'][tio.DATA]#.permute(0,1,3,2)

    # preds.append(pred)
    # gts.append(gt)




    preds = pred.numpy()
    gts = gt.numpy()



    pred = preds.astype(int)  # float data does not support bit_and and bit_or
    gdth = gts.astype(int)  # float data does not support bit_and and bit_or
    fp_array = copy.deepcopy(pred)  # keep pred unchanged
    fn_array = copy.deepcopy(gdth)
    gdth_sum = np.sum(gdth)
    pred_sum = np.sum(pred)
    intersection = gdth & pred
    union = gdth | pred
    intersection_sum = np.count_nonzero(intersection)
    union_sum = np.count_nonzero(union)

    tp_array = intersection

    tmp = pred - gdth
    fp_array[tmp < 1] = 0

    tmp2 = gdth - pred
    fn_array[tmp2 < 1] = 0

    tn_array = np.ones(gdth.shape) - union

    tp, fp, fn, tn = np.sum(tp_array), np.sum(fp_array), np.sum(fn_array), np.sum(tn_array)

    smooth = 0.001
    precision = tp / (pred_sum + smooth)
    recall = tp / (gdth_sum + smooth)

    false_positive_rate = fp / (fp + tn + smooth)
    false_negtive_rate = fn / (fn + tp + smooth)
    acc = (tp+tn) / (tp+fp+fn+tn)

    jaccard = intersection_sum / (union_sum + smooth)
    dice = 2 * intersection_sum / (gdth_sum + pred_sum + smooth)
    sen = tp/(tp+fn+smooth)
    spe = tn/(tn+fp+smooth)  
    # print(false_positive_rate)
    # print(false_negtive_rate)
    # print(precision)
    # print(recall)
    # print(dice)

    acc_summary.append(acc)
    pre_summary.append(precision)
    rec_summary.append(recall)
    dice_summary.append(dice)
 

acc_mean = np.mean(acc_summary)
pre_mean = np.mean(pre_summary)
rec_mean = np.mean(rec_summary)
dice_mean = np.mean(dice_summary)


acc_sted = np.std(acc_summary)
pre_sted = np.std(pre_summary)
rec_sted = np.std(rec_summary)
dice_sted = np.std(dice_summary)

# print(acc_mean)
# print(acc_sted)
print(pre_mean)
print(pre_sted)
print(rec_mean)
print(rec_sted)
print(dice_mean)
print(dice_sted)
