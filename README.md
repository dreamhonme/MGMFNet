# Mask-Guided Multi-Scale Fusion Network for Skin Lesion Segmentation
# Abstract
<h4>Precise segmentation of skin lesions is essential for accurate diagnosis and effective treatment. However, existing segmentation approaches struggle with lesion heterogeneity, scale variability, indistinct boundaries, and low image contrast. Conventional neural networks are constrained by their limited receptive fields and fixed kernel sizes, hindering their ability to capture long-range spatial dependencies. In addition, key spatial details are frequently lost during feature pyramid construction, leading to suboptimal detection of small, low-contrast lesions. To overcome these limitations, the Mask-Guided Multi-Scale Fusion Network (MGMFNet) is proposed, which incorporates two novel modules. First, the Mask-Guided Multi-Scale Convolution Fusion decoder (MGMSCF) leverages predicted masks to generate multi-scale focus regions, enabling the model to hierarchically localize lesion features and fuse features in a context-aware manner. This design preserves critical details and enhances the model’s adaptability to varying lesion shapes. Second, the Dual-Path Parallel Adaptive Attention Mechanism (DPAM) establishes bidirectional adaptive interactions between shallow spatial and deep semantic features, mitigating information degradation during feature propagation. Experimental results on the ISIC-2016, ISIC-2017, and ISIC-2018 datasets demonstrate that MGMFNet achieves superior performance in edge preservation and fine-detail segmentation. These results highlight its strong generalization capability and potential for practical clinical applications.https://github.com/dreamhonme/MGMFNet</h4>
All results is evaluated on Python 3.10 with PyTorch 2.1.2+cuda121.We publish our test results on the ISIC2018,ISIC2017,ISIC2016,ph2 and CVC-ClinicDB.
# Datasets
The ISIC2018, ISIC2017,ISIC2016,ph2 and CVC-ClinicDB datasets can be downloaded with:
<h4></h4>https://challenge.isic-archive.com/data/#2018</h4>
## https://challenge.isic-archive.com/data/#2017
## https://challenge.isic-archive.com/data/#2016
## https://www.fc.up.pt/addi/ph2%20database.html
## https://tianchi.aliyun.com/dataset/93690
