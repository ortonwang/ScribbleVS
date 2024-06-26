import argparse
import os
import re
import shutil
import warnings
warnings.filterwarnings('ignore')
import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm

# from networks.efficientunet import UNet
from networks.net_factory import net_factory

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--ori_img_path', type=str, default='./data/ACDC_training/', help='experiment_name')
# parser.add_argument('--root_path', type=str,default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='ACDC/debug', help='experiment_name')
parser.add_argument('--model', type=str, default='unet', help='model_name,unet_cct,unet')
parser.add_argument('--fold', type=str,default='MAAGfold', help='fold')
parser.add_argument('--num_classes', type=int,  default=4, help='output channel of network')
parser.add_argument('--sup_type', type=str, default="scribble",help='label,scribble')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
FLAGS = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
def get_fold_ids(fold):
    all_cases_set = ["patient{:0>3}".format(i) for i in range(1, 101)]
    fold1_testing_set = [
        "patient{:0>3}".format(i) for i in range(1, 21)]
    fold1_training_set = [
        i for i in all_cases_set if i not in fold1_testing_set]

    fold2_testing_set = [
        "patient{:0>3}".format(i) for i in range(21, 41)]
    fold2_training_set = [
        i for i in all_cases_set if i not in fold2_testing_set]

    fold3_testing_set = [
        "patient{:0>3}".format(i) for i in range(41, 61)]
    fold3_training_set = [
        i for i in all_cases_set if i not in fold3_testing_set]

    fold4_testing_set = [
        "patient{:0>3}".format(i) for i in range(61, 81)]
    fold4_training_set = [
        i for i in all_cases_set if i not in fold4_testing_set]

    fold5_testing_set = [
        "patient{:0>3}".format(i) for i in range(81, 101)]
    fold5_training_set = [
        i for i in all_cases_set if i not in fold5_testing_set]
    if fold == "fold1":
        return [fold1_training_set, fold1_testing_set]
    elif fold == "fold2":
        return [fold2_training_set, fold2_testing_set]
    elif fold == "fold3":
        return [fold3_training_set, fold3_testing_set]
    elif fold == "fold4":
        return [fold4_training_set, fold4_testing_set]
    elif fold == "fold5":
        return [fold5_training_set, fold5_testing_set]
    elif "MAAGfold" in fold:
        training_set = ["patient{:0>3}".format(i) for i in
                        [37, 50, 53, 100, 38, 19, 61, 74, 97, 31, 91, 35, 56, 94, 26, 69, 46, 59, 4, 89,
                         71, 6, 52, 43, 45, 63, 93, 14, 98, 88, 21, 28, 99, 54, 90, 2, 76, 34, 85, 70, 86, 3, 8, 51, 40,
                         7, 13, 47, 55, 12, 58, 87, 9, 65, 62, 33, 42,
                         23, 92, 29, 11, 83, 68, 75, 67, 16, 48, 66, 20, 15]]
        testing_set = ["patient{:0>3}".format(i) for i in
                          [5, 39, 77, 82, 78, 10, 64, 24, 30, 73, 80, 41, 36, 60, 72]]
        return [training_set, testing_set]
    else:
        return "ERROR KEY"


def calculate_metric_percase(pred, gt, spacing):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    asd = metric.binary.asd(pred, gt, voxelspacing=spacing)
    hd95 = metric.binary.hd95(pred, gt, voxelspacing=spacing)
    # hd95,asd = 0,0
    return dice, hd95, asd


def test_single_volume(case, net, test_save_path, FLAGS):
    h5f = h5py.File(FLAGS.root_path +
                    "/ACDC_training_volumes/{}".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction = np.zeros_like(label)
    size = 256
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (size / x, size / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out_main = net(input)#[0]
            out = torch.argmax(torch.softmax(
                out_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / size, y / size), order=0)
            prediction[ind] = pred
    case = case.replace(".h5", "")
    org_img_path = FLAGS.org_img_path  + "/{}.nii.gz".format(case)
    org_img_itk = sitk.ReadImage(org_img_path)
    spacing = org_img_itk.GetSpacing()

    first_metric = calculate_metric_percase(
        prediction == 1, label == 1, (spacing[2], spacing[0], spacing[1]))
    second_metric = calculate_metric_percase(
        prediction == 2, label == 2, (spacing[2], spacing[0], spacing[1]))
    third_metric = calculate_metric_percase(
        prediction == 3, label == 3, (spacing[2], spacing[0], spacing[1]))

    # img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    # img_itk.CopyInformation(org_img_itk)
    # prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    # prd_itk.CopyInformation(org_img_itk)
    # lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    # lab_itk.CopyInformation(org_img_itk)
    # sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
    # sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
    # sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
    return first_metric, second_metric, third_metric


def Inference(FLAGS):
    train_ids, test_ids = get_fold_ids(FLAGS.fold)
    all_volumes = os.listdir(
        FLAGS.root_path + "/ACDC_training_volumes")
    image_list = []
    for ids in test_ids:
        new_data_list = list(filter(lambda x: re.match(
            '{}.*'.format(ids), x) != None, all_volumes))
        image_list.extend(new_data_list)
    snapshot_path = "../model/{}_{}/{}".format(
        FLAGS.exp, FLAGS.fold, FLAGS.sup_type)
    test_save_path = "../model/{}_{}/{}/{}_predictions/".format(
        FLAGS.exp, FLAGS.fold, FLAGS.sup_type, FLAGS.model)

    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    net = net_factory(net_type=FLAGS.model, in_chns=1,
                      class_num=FLAGS.num_classes)
    save_mode_path = os.path.join(
        snapshot_path, 'unet_best_model.pth')

    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()

    first_total = 0.0
    second_total = 0.0
    third_total = 0.0
    for case in tqdm(image_list):
        print(case)
        first_metric, second_metric, third_metric = test_single_volume(
            case, net, test_save_path, FLAGS)
        first_total += np.asarray(first_metric)
        second_total += np.asarray(second_metric)
        third_total += np.asarray(third_metric)
    avg_metric = [first_total / len(image_list), second_total /
                  len(image_list), third_total / len(image_list)]
    print(avg_metric)
    print((avg_metric[0] + avg_metric[1] + avg_metric[2]) / 3)
    return ((avg_metric[0] + avg_metric[1] + avg_metric[2]) / 3)[0]

if __name__ == '__main__':

    mean_dice = Inference(FLAGS)
    print(mean_dice)
