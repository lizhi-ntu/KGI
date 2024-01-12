# Virtual Try-On with Garment-Pose Keypoints Guided Inpainting

WIP

This codes repository provides the pytorch implementation of the KGI virtual try-on method proposed in ICCV23 paper [Virtual Try-On with Garment-Pose Keypoints Guided Inpainting.](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Virtual_Try-On_with_Pose-Garment_Keypoints_Guided_Inpainting_ICCV_2023_paper.pdf)

## Experimental Environment 
Please follow the steps below to build the environment and install the required packages.
```
conda create -n kgi python=3.8 -y
conda activate kgi
bash install_pkgs.sh
```

## Data Preparation
1. The VITON-HD dataset could be downloaded from [VITON-HD](https://www.dropbox.com/s/10bfat0kg4si1bu/zalando-hd-resized.zip?dl=0) . Please place the dataset under the directory `KGI/data/`. The dataset contains the following content:
   |Content|Comment|
   |-------|-------|
   |agnostic-mask|not in use in KGI|
   |agnostic-v3.2|not in use in KGI|
   |cloth ||
   |cloth-mask|not in use in KGI|
   |image||
   |image-densepose|not in use in KGI|
   |image-parse-agnostic-v3.2|not in use in KGI|
   |image-parse-v3|
   |openpose_img|not in use in KGI|
   |openpose_json|
   
2. In addition to above content, some other preprocessed conditions are in use in KGI. The content are generated with the data preprocessing codes [WIP]. The preprocessed data could be downloaded, respectively.
   |Content|Train|Test|
   |---|---|---|
   |image-landmark-json|[Google Drive](https://drive.google.com/file/d/1G02Vo93laqDcPAD2_AUufoa_IOWKOerK/view?usp=drive_link)|[Google Drive](https://drive.google.com/file/d/1y1GTxQGTL57lvFXDko3pZgDd6Ag3ER3O/view?usp=drive_link)|
   |cloth-landmark-json|[Google Drive](https://drive.google.com/file/d/1QgEBXEm-md6nus0jAV7IXGU1KjHNqX4o/view?usp=drive_link)|[Google Drive](https://drive.google.com/file/d/1TCv4BzzjwrnLnfJrw_3MSnzzMKyr0dV4/view?usp=drive_link)|
   |label|[Google Drive](https://drive.google.com/file/d/1lhOOET1yREmvTyMjQRzEkWRuIw8zcYIG/view?usp=drive_link)|[Google Drive](https://drive.google.com/file/d/1SKTF9EztKlb3NXl0VhcoKjbU6ILUq1J8/view?usp=drive_link)|
   |parse|[Google Drive](https://drive.google.com/file/d/1kzRjJSFtDouCdOcXK762firAKJHf4Mb-/view?usp=drive_link)|[Google Drive](https://drive.google.com/file/d/1ponGgA-dVg2z71ebFA4W-1ndGsqV2Xwl/view?usp=drive_link)|
   |parse_ag_full|[Google Drive](https://drive.google.com/file/d/1zXYHM9MmwEDi9zAeqLBXWh_xZK18-Wvj/view?usp=drive_link)|[Google Drive](https://drive.google.com/file/d/1DgtU6TQJC7hPMu6kC8R4NsjH4Bb2kU9j/view?usp=drive_link)|
   |ag_mask|[Google Drive](https://drive.google.com/file/d/1m-d6_YkhLV6Xjg06yBGVVCQ3COFxmrLv/view?usp=drive_link)|[Google Drive](https://drive.google.com/file/d/1KLW2XERt5BlEcnbjzJ-tVFzfTO3oNgN-/view?usp=drive_link)|
   |skin_mask|[Google Drive](https://drive.google.com/file/d/1wSD4VH-auZaNIbY87whR5gLB6l-MF20d/view?usp=drive_link)|[Google Drive](https://drive.google.com/file/d/1E09kBvCNVZdXxLg0Dx7SNM4PlKWkzkGs/view?usp=drive_link)|
   
   
## Model Training
WIP
## Demo with Pretrained Model
WIP
