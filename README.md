# jsws

## Environment
clone repo:
```
git clone https://github.com/zengxianyu/jsws.git
git submodule init 
git submodule update
```

prepare environment:
```
conda env create --file=environments.yaml
```

## Prepare Data
[DUTS-train (Onedrive)](https://maildluteducn-my.sharepoint.com/:u:/g/personal/zengyu_mail_dlut_edu_cn/EaJni8OcXzxJi1BDQsjqh4YBFlY_UlMNHvF6TGm43dIDWg?e=AhNHVk)

[ECSSD (Onedrive)](https://maildluteducn-my.sharepoint.com/:u:/g/personal/zengyu_mail_dlut_edu_cn/EcLF1rbjDY9AvWmZ0mui9owB-l3t0zo270d1aK6E_Crp2w?e=fZME4U)

[VOC2012 (Onedrive)](https://maildluteducn-my.sharepoint.com/:u:/g/personal/zengyu_mail_dlut_edu_cn/EVUJBg67ICxHqB_wfehc34gBQKi_RTJgnTCcUPnwxfTSIA?e=ef0AJw)

[SegmentationClassAug (Onedrive)](https://maildluteducn-my.sharepoint.com/:u:/g/personal/zengyu_mail_dlut_edu_cn/EXhmcGsGEaBPnhOffoNlh2UBUyZuB7Eck5WUbJ3f3pSSbA?e=vLLc34)

## Train stage 1
train using image-level class labels and saliency ground-truth:

```shell
weak_seg_full_sal_train.py
```

Open the file output/logs/train1.html in browser for visualization

It should be easy to achieve MIOU>54 but you may need to try multiple times to get the score MIOU 57.1 or more than that in Table. 5 of the paper. 

## Train stage 2
train a more complex model using the prediction of the model trained in the stage 1. 

```shell
syn_stage1.py
self_seg_full_sal_train.py
```

## Saliency results

[download saliency maps (Google Drive)](https://drive.google.com/open?id=1KqO8bhJn2StXGblBL_9V6-yM2CSOBNsz); [One Drive](https://1drv.ms/u/s!AqVkBGUQ01XGjxiqc5pdH20yPXz4?e=WzCpBW)

## Citation
```
@inproceedings{zeng2019joint,
  title={Joint learning of saliency detection and weakly supervised semantic segmentation},
  author={Zeng, Yu and Zhuge, Yunzhi and Lu, Huchuan and Zhang, Lihe},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  year={2019}
}
```
