# Action Segmentation with Joint Self-Supervised Temporal Domain Adaptation

---
This is the official repo of our paper:

**Action Segmentation with Joint Self-Supervised Temporal Domain Adaptation**  
[__***Min-Hung Chen***__](https://www.linkedin.com/in/chensteven), [Baopu Li](https://www.linkedin.com/in/paul-lee-46b2382b/), [Yingze Bao](https://www.linkedin.com/in/yingze/), [Zsolt Kira](https://www.cc.gatech.edu/~zk15/), [Ghassan AlRegib (Advisor)](https://ghassanalregib.info/) <br>
[IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020](http://cvpr2020.thecvf.com/)   
[arXiv (coming soon!)]

<p align="center">
<img src="webpage/Overview.png?raw=true" width="70%">
</p>

Despite the recent progress of fully-supervised action segmentation techniques, the performance is still not fully satisfactory. One main challenge is the problem of spatiotemporal variations (e.g. different people may perform the same activity in various ways). Therefore, we exploit unlabeled videos to address this problem by reformulating the action segmentation task as a cross-domain problem with domain discrepancy caused by spatio-temporal variations. To reduce the discrepancy, we propose **Self-Supervised Temporal Domain Adaptation (SSTDA)**, which contains two self-supervised auxiliary tasks (binary and sequential domain prediction) to jointly align cross-domain feature spaces embedded with local and global temporal dynamics, achieving better performance than other Domain Adaptation (DA) approaches. On three challenging benchmark datasets (GTEA, 50Salads, and Breakfast), SSTDA outperforms the current state-of-the-art method by large margins (e.g. for the F1@25 score, from 59.6% to 69.1% on Breakfast, from 73.4% to 81.5% on 50Salads, and from 83.6% to 89.1% on GTEA), and requires only 65% of the labeled training data for comparable performance, demonstrating the usefulness of adapting to unlabeled target videos across variations.

---

(Stay tuned for the code release)

---
### Contact
[Min-Hung Chen](https://www.linkedin.com/in/chensteven) <br>
cmhungsteve AT gatech DOT edu <br>
[<img align="left" src="webpage/OLIVES_new.png" width="15%">](https://ghassanalregib.info/)
