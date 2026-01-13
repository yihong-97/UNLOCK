<p align="center">
<h1 align="center"><strong>Unlocking Constraints: Source-Free Occlusion-Aware Seamless Segmentation</strong></h1>
<h3 align="center">ICCV 2025</h3>

<p align="center">
    <a href="https://github.com/yihong-97">Yihong Cao</a><sup>1,2*</sup>,</span>
    <a href="https://jamycheung.github.io">Jiaming Zhang</a><sup>3,4*</sup>,
    <a href="https://zhengxujosh.github.io/">Xu Zheng</a><sup>5,6</sup>,
    <a href="https://github.com/MasterHow">Hao Shi</a><sup>7</sup>,
    <a href="https://github.com/KPeng9510">Kunyu Peng</a><sup>2</sup>,
    <a>Hang Liu</a><sup>1</sup>,
    <a href="https://yangkailun.com">Kailun Yang</a><sup>1†</sup>,
    <a href="http://robotics.hnu.edu.cn/info/1176/2966.htm">Hui Zhang</a><sup>1†</sup>
    <br>
        <sup>1</sup>Hunan University,
        <sup>2</sup>Hunan Normal University,
        <sup>3</sup>Karlsruhe Institute of Technology,
        <sup>4</sup>ETH Zurich,
        <sup>5</sup>HKUST(GZ),
        <sup>6</sup>INSAIT, Sofia University “St. Kliment Ohridski”,
        <sup>7</sup>Zhejiang University
</p>



## Abstract
Panoramic image processing is essential for omni-context perception, yet faces constraints like distortions, perspective occlusions, and limited annotations. Previous unsupervised domain adaptation methods transfer knowledge from labeled pinhole data to unlabeled panoramic images, but they require access to source pinhole data. To address these, we introduce a more practical task, _ie_, Source-Free Occlusion-Aware Seamless Segmentation (SFOASS), and propose its first solution, called **UNconstrained Learning Omni-Context Knowledge (UNLOCK)**. Specifically, UNLOCK includes two key modules: Omni Pseudo-Labeling Learning and Amodal-Driven Context Learning. While adapting without relying on source data or target labels, this framework enhances models to achieve segmentation with 360° viewpoint coverage and occlusion-aware reasoning. Furthermore, we benchmark the proposed SFOASS task through both real-to-real and synthetic-to-real adaptation settings. Experimental results show that our source-free method achieves performance comparable to source-dependent methods, yielding state-of-the-art scores of 10.9 in mAAP and 11.6 in mAP, along with an absolute improvement of +4.3 in mAPQ over the source-only method.

<p align="center">
<img src="./fig/banner.png" width="1080px"/>  
<br>
<em>UNLOCK framework solves the Source-Free Occlusion-Aware Seamless Segmentation (SFOASS), enabling segmentation with 360° viewpoint coverage and occlusion-aware reasoning while adapting without requiring source data and target labels</em>
</p>

<div align="center">

[[PDF]](https://openaccess.thecvf.com/content/ICCV2025/papers/Cao_Unlocking_Constraints_Source-Free_Occlusion-Aware_Seamless_Segmentation_ICCV_2025_paper.pdf)  [[Supp]](https://openaccess.thecvf.com/content/ICCV2025/papers/Cao_Unlocking_Constraints_Source-Free_Occlusion-Aware_Seamless_Segmentation_ICCV_2025_paper.pdf)  [[arXiv]](https://arxiv.org/pdf/2506.21198)

</div>

### 📚 Datasets

This work addresses **Source-Free Occlusion-Aware Seamless Segmentation (SFOASS)** and evaluates the proposed method under two domain adaptation settings. In both cases, the **source domains remain unchanged**, and we **convert the target dataset (BlendPASS) to align with the source label space**.

#### 1. Real-to-Real Adaptation  
- **Source**: [**KITTI-360 APS**](https://amodal-panoptic.cs.uni-freiburg.de/) — a real-world amodal panoptic dataset.  
- **Target**: [**BlendPASS**](https://github.com/yihong-97/BlendPASS) — a real-world 360° street-view panoptic segmentation dataset.

#### 2. Synthetic-to-Real Adaptation  
- **Source**: [**AmodalSynthDrive**](https://amodalsynthdrive.cs.uni-freiburg.de/) — a synthetic dataset for amodal panoptic segmentation in driving scenes.  
- **Target**: [**BlendPASS**](https://github.com/yihong-97/BlendPASS) — same as above.

#### 🔗 Download Converted BlendPASS Datasets
We provide two versions of **BlendPASS**, each aligned to the respective source domain’s **7-class label space**:

| Source Domain       | Converted Target Dataset | Download Link |
|---------------------|--------------------------|---------------|
| KITTI-360 APS       | BlendPASS (APS-aligned)  | [Google Drive](https://drive.google.com/drive/folders/1T0GDze9s-aQw86PFzrKCSkxx4Un_Kd-A?usp=sharing) |
| AmodalSynthDrive    | BlendPASS (ASD-aligned)  | [Google Drive](https://drive.google.com/drive/folders/1kGSEHhBHGeRJEtfpYHXye3N3Z8z3eHH5?usp=sharing) |

#### 🔗 Download Converted BlendPASS Datasets

We provide two versions of **BlendPASS**, each aligned to the respective source domain’s **7-class label space**. Additionally, we release the preprocessed training labels for both source datasets used in our experiments.

| Dataset                     | Description                                | Download Link |
|----------------------------|--------------------------------------------|---------------|
| **BlendPASS (APS-aligned)**   | Target dataset aligned to KITTI-360 APS     | [Google Drive](https://drive.google.com/drive/folders/1T0GDze9s-aQw86PFzrKCSkxx4Un_Kd-A?usp=sharing) |
| **BlendPASS (ASD-aligned)**   | Target dataset aligned to AmodalSynthDrive  | [Google Drive](https://drive.google.com/drive/folders/1kGSEHhBHGeRJEtfpYHXye3N3Z8z3eHH5?usp=sharing) |
| **KITTI-360 APS Labels**      | Preprocessed amodal panoptic training labels   | [Google Drive]() |
| **AmodalSynthDrive Labels**   | Preprocessed amodal panoptic training labels   | [Google Drive]() |


## Contact
If you have any suggestions or find our work helpful, feel free to contact us

Email: caoyihong97@foxmail.com

If you find our work useful, please consider citing it:

```
@InProceedings{Cao_2025_ICCV,
    author    = {Cao, Yihong and Zhang, Jiaming and Zheng, Xu and Shi, Hao and Peng, Kunyu and Liu, Hang and Yang, Kailun and Zhang, Hui},
    title     = {Unlocking Constraints: Source-Free Occlusion-Aware Seamless Segmentation},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {8961-8972}
}
```


# run /configs/SFDA_unmaskformer/Generate_pseudolabels_sourcemodel.py to generate the results with numpy format
# run /tools/convert_pseudolabels/Save_levelinstance_softlabels.py (Line 146) and /tools/convert_pseudolabels/Save_semantic_softlabels.py (Line 130)
# Then, rename these three directories to 'semantic' 'instance' and 'amodal_instance'
# run configs/SFDA_unmaskformer/Training_targetonly_unmaskformer.py (Line 75)
# The best results will be obtained in 400 iters