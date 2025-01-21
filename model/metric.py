import torch
import torch.nn.functional as F
from sklearn.metrics import *
from scipy import interpolate

def _ece_score(y_true,y_pred,bins=10):
    ece = 0.
    for i in range(bins):
        c_start,c_end = i/bins,(i+1)/bins
        mask = (c_start<=y_pred)&(y_pred<c_end)
        ni = mask.count_nonzero().item()
        if ni==0:
            continue
        acc,conf = y_true[mask].sum()/ni,y_pred[mask].mean()
        ece += ni*(acc-conf).abs()
    return float(ece)/len(y_true)

def ACC(correct, target, region):
    acc = correct.sum().item()/len(target)

    # map each sample to its class region
    region_map = torch.zeros(len(target),dtype=torch.int64)
    region_map[region[0]:region[0]+region[1]] = 1
    region_map[region[0]+region[1]:sum(region)] = 2

    # count correct samples in each region
    split_acc, num_samples = [0, 0, 0], [0, 0, 0]
    for i in range(len(target)):
        split_acc[region_map[target[i].item()]] += correct[i].item()
        num_samples[region_map[target[i].item()]] += 1
    split_acc = [split_acc[i]/num_samples[i] for i in range(3)]

    print('ACC (%):')
    print(f'\tall  = {acc*100:.2f}')
    print(f'\thead = {split_acc[0]*100:.2f}')
    print(f'\tmed  = {split_acc[1]*100:.2f}')
    print(f'\ttail = {split_acc[2]*100:.2f}')

def AUCECE(correct, uncertainty, target, region):
    confidence = 1 - uncertainty
    auc = roc_auc_score(correct, confidence)
    ece = _ece_score(correct, confidence)

    # map each sample to its class region
    region_map = torch.zeros(len(target), dtype=torch.int64)
    region_map[region[0] : region[0] + region[1]] = 1
    region_map[region[0] + region[1] : sum(region)] = 2

    # count correct samples in each region
    split_auc, split_ece = [0, 0, 0], [0, 0, 0]
    for i in range(3):
        mask = (region_map[target] == i)
        split_auc[i] = roc_auc_score(correct[mask], confidence[mask])
        split_ece[i] = _ece_score(correct[mask], confidence[mask])

    print('AUC (%):')
    print(f'\tall  = {auc*100:.2f}')
    print(f'\thead = {split_auc[0]*100:.2f}')
    print(f'\tmed  = {split_auc[1]*100:.2f}')
    print(f'\ttail = {split_auc[2]*100:.2f}')

    print('ECE (%):')
    print(f'\tall  = {ece*100:.2f}')
    print(f'\thead = {split_ece[0]*100:.2f}')
    print(f'\tmed  = {split_ece[1]*100:.2f}')
    print(f'\ttail = {split_ece[2]*100:.2f}')

def FHR(pred,target,num_class):
    # tail ratio: 25% 50% 75% avg
    tail_rate = [0.25,0.50,0.75]
    fhr = []
    for rate in tail_rate:
        border_cls = int(num_class*rate+0.5)
        num_tail_target = torch.count_nonzero(border_cls<=target).item()
        num_false_head = torch.count_nonzero((border_cls<=target)&(pred<border_cls)).item()
        fhr.append(num_false_head/num_tail_target)

    print(f'FHR (%):')
    print(f'\t0.25 = {fhr[0]*100:.2f}')
    print(f'\t0.5  = {fhr[1]*100:.2f}')
    print(f'\t0.75 = {fhr[2]*100:.2f}')
    print(f'\tavg  = {sum(fhr)/len(fhr)*100:.2f}')

def CIAFR10_bird2plane(pred,target):
    mask = (target==2)
    mask_err = mask&(pred==0)
    print(f'Bird2Plane (%): {torch.count_nonzero(mask_err).item()/torch.count_nonzero(mask).item()*100:.2f}')

def CIAFR10_car2animal(pred,target):
    mask = (target==1)|(target==9)
    mask_err = mask&(pred>=3)&(pred<8)
    print(f'Car2Animal (%): {torch.count_nonzero(mask_err).item()/torch.count_nonzero(mask).item()*100:.2f}')
