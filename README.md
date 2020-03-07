# Action Segmentation with Joint Self-Supervised Temporal Domain Adaptation

This is the official PyTorch implementation of our paper:

**Action Segmentation with Joint Self-Supervised Temporal Domain Adaptation**  
[__***Min-Hung Chen***__](https://www.linkedin.com/in/chensteven), [Baopu Li](https://www.linkedin.com/in/paul-lee-46b2382b/), [Yingze Bao](https://www.linkedin.com/in/yingze/), [Zsolt Kira](https://www.cc.gatech.edu/~zk15/), [Ghassan AlRegib (Advisor)](https://ghassanalregib.info/) <br>
[*IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2020*](http://cvpr2020.thecvf.com/)   
[[arXiv](https://arxiv.org/abs/2003.02824)]

<p align="center">
<img src="webpage/Overview.png?raw=true" width="70%">
</p>

Despite the recent progress of fully-supervised action segmentation techniques, the performance is still not fully satisfactory. One main challenge is the problem of spatiotemporal variations (e.g. different people may perform the same activity in various ways). Therefore, we exploit unlabeled videos to address this problem by reformulating the action segmentation task as a cross-domain problem with domain discrepancy caused by spatio-temporal variations. To reduce the discrepancy, we propose **Self-Supervised Temporal Domain Adaptation (SSTDA)**, which contains two self-supervised auxiliary tasks (binary and sequential domain prediction) to jointly align cross-domain feature spaces embedded with local and global temporal dynamics, achieving better performance than other Domain Adaptation (DA) approaches. On three challenging benchmark datasets (GTEA, 50Salads, and Breakfast), SSTDA outperforms the current state-of-the-art method by large margins (e.g. for the F1@25 score, from 59.6% to 69.1% on Breakfast, from 73.4% to 81.5% on 50Salads, and from 83.6% to 89.1% on GTEA), and requires only 65% of the labeled training data for comparable performance, demonstrating the usefulness of adapting to unlabeled target videos across variations.

---
## Requirements
Tested with:
* Ubuntu 18.04.2 LTS
* PyTorch 1.1.0
* Torchvision 0.3.0
* Python 3.7.3
* GeForce GTX 1080Ti
* CUDA 9.2.88
* CuDNN 7.14

Or you can directly use our environment file:
```
conda env create -f environment.yml
```

---
## Data Preparation
* Clone the this repository:
```
git clone https://gitlab.svail.baidu.com/steve/action-segmentation-DA.git
cd action-segmentation-DA
```
* Download the [Dataset](https://www.dropbox.com/s/yodx2dknti0ah2v/Datasets.zip?dl=0) folder, which contains the features and the ground truth labels. (~30GB)
* Extract it so that you have the `Datasets` folder.
* The default path for the dataset is `../../Datasets/action-segmentation/` if the current location is `./action-segmentation-DA/`. If you change the dataset path, you need to edit the scripts as well.

---
## Usage
#### Quick Run
* Since there are lots of arguments, we recommend to directly run the scripts.
* All the scripts are in the folder `scripts/` with the name `run_<dataset>_<method>.sh`.
* You can simply copy any script to the main folder (same location as all the `.py` files), and run the script as below:
```
# one example
bash run_gtea_SSTDA.sh
```
The script will do training, predicting and evaluation for all the splits on the dataset (`<dataset>`) using the method (`<method>`).

#### More Details
* In each script, you may want to modify the following sections:
  * `# === Mode Switch On/Off === #`
    * `training`, `predict` and `eval` are the modes that can be switched on or off by set as `true` or `false`.
  * `# === Paths === #`
    * `path_data` needs to be the same as the location of the input data.
    * `path_model` and `path_result` are the path for output models and prediction. The folders will be created if not existing.
  * `# === Main Program === #`
    * You can run only the specific splits by editing `for split in 1 2 3 ...` (line 53).
* We DO NOT recommend to edit other parts (e.g. `# === Config & Setting === #
`); otherwise the implementation may be different.

---
## Citation
If you find this repository useful, please cite our paper:
```
@inproceedings{chen2020action,
  title={Action Segmentation with Joint Self-Supervised Temporal Domain Adaptation},
  author={Chen, Min-Hung and Li, Baopu and Bao, Yingze and AlRegib, Ghassan and Kira, Zsolt},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2020},
  url={https://arxiv.org/abs/2003.02824}
}
```

---
### Acknowledgments
This work was done with the support from [**OLIVES**](https://ghassanalregib.info/)@GT. <br>
Feel free to check our lab's [Website](https://ghassanalregib.info/) and [GitHub](https://github.com/olivesgatech) for other interesting work!!!


Some codes are borrowed from [ms-tcn](https://github.com/yabufarha/ms-tcn), [TA3N](https://github.com/cmhungsteve/TA3N), [swd_pytorch](https://github.com/krumo/swd_pytorch), and [VCOP](https://github.com/xudejing/VCOP).


---
### Contact
[Min-Hung Chen](https://www.linkedin.com/in/chensteven) <br>
cmhungsteve AT gatech DOT edu <br>
[<img align="left" src="webpage/OLIVES_new.png" width="15%">](https://ghassanalregib.info/)
