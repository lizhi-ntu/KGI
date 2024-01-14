# Virtual Try-On with Garment-Pose Keypoints Guided Inpainting

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
   * Data Preprocessing [WIP]
3. Download the [demo_paired_pairs.txt](https://drive.google.com/file/d/1wcMj7S-P6XnyePrpa_VgZpN5mXpDJLRA/view?usp=drive_link) and [demo_unpaired_pairs.txt](https://drive.google.com/file/d/12dB0Zh5iVZmVz7ptss3Lw5Y6zl0s1EQ0/view?usp=drive_link) under the directory `KGI/data/zalando-hd-resized/` for in-training visualization.
4. The structure of processed dataset should be as below:
   * KGI/data/zalando-hd-resized/
      * test/
         * ag_mask/
         * cloth/
         * cloth-landmark-json/
         * image/
         * image-landmark-json/
         * image-parse-v3/
         * openpose_json/
         * parse/
         * parse_ag_full/
         * skin_mask/
         * label.json
      * train/
         * ...
      * demo_paired_pairs.txt
      * demo_unpaired_pairs.txt
      * test_pairs.txt
      * train_pairs.txt
   
## Model Training
The model training of the KGI method consists of three steps: Training of the Keypoints Generator, Training of the Parse Generator, Training of the Semantic Conditioned Inpainting Model.
### Keypoints Generator
* The Keypoints Generator is trained with the following scripts:
   ```
   cd codes_kg
   python3 train_kg.py
   ```
  During the training, the visualization of some validation samples will be saved under directory `KGI/visualizations/two_graph_cs/`. Below is an example of visualization results.
  ![Demo Image 1](https://github.com/lizhi-ntu/KGI/blob/main/imgs/demo_img1.jpg)
  The pretrianed checkpoints of Keypoints Generator could be downloaded from [Google Drive](https://drive.google.com/file/d/1FQbeWkcqgSqycznGW6INWt2kkrk5nG3H/view?usp=drive_link) and put under the directory `KGI/checkpoints_pretrained/kg/`.
* Since the parse generation is based on the estimated keypoints conditions, please generate the keypoints conditions with the following scripts before the training of Parse Generator:
  ```
  bash generate_kg_demo_paired.sh
  bash generate_kg_demo_unpaired.sh
  bash generate_kg_train.sh
  bash generate_kg_test_paired.sh
  bash generate_kg_test_unpaired.sh
  ```
  The generated keypoints conditions will be saved under the directory `KGI/example/generate_kg/` and also could be downloaded from [train_kg_conditions](https://drive.google.com/file/d/1Fpte42AvLOgnXtaiND9J3K2Q9DpIsLQe/view?usp=drive_link) and [test_kg_conditions](https://drive.google.com/file/d/1JAmDUJRYFvSsIperVtvWqIjdod6loruM/view?usp=drive_link). The files should be placed under the directories `KGI/data/zalando-hd-resized/train/` and `KGI/data/zalando-hd-resized/test/`, respectively.
### Parse Generator
* The Parse Generator is trained with the following scripts:
  ```
  cd codes_pg
  python3 train_pg.py
  ```
  During the training, the visualization of some validation samples will be saved under directory `KGI/visualizations/parse_full/`. Below is an example of visualization results.
  ![Demo Image 2](https://github.com/lizhi-ntu/KGI/blob/main/imgs/demo_img2.jpg)
  The pretrianed checkpoints of Parse Generator could be downloaded from [Google Drive](https://drive.google.com/file/d/1_2lTeLdCczBnnaXsMxBtvbLZv2GY3PCp/view?usp=drive_link) and put under the directory `KGI/checkpoints_pretrained/pg/`.
* After the training of the Parse Generator, the person image parse (estimated segmentation map) could be generated with the following scripts:
  ```
  bash generate_pg_demo_paired.sh
  bash generate_pg_demo_unpaired.sh
  bash generate_pg_test_paired.sh
  bash generate_pg_test_unpaired.sh
  ```
  The generated parse conditions will be saved under the directory `KGI/example/generate_pg/` and also could be downloaded from [test_pg_conditions](https://drive.google.com/file/d/1Zq-v9BzyxOM_b_1EKQNz0ks8vF-ixpyw/view?usp=drive_link). The files should be placed under the directory `KGI/data/zalando-hd-resized/test/` for tps conditions and final results generation. 
### Semantic Conditioned Inpainting Model
* The Semantic Conditioned Inpainting Model is trained with the following scripts:
  ```
  cd codes_sdm
  python3 train_sdm.py
  ```
  The pretrianed checkpoints of Semantic Conditioned Inpainting Model could be downloaded from [Google Drive](https://drive.google.com/file/d/1guxm63mRMH64xiycDHsEqNb_q3KqueDB/view?usp=drive_link) and put under the directory `KGI/checkpoints_pretrained/sci/ckpt_1024/`.
* The tps conditions (recomposed person image and content keeping mask) could be generated with the following scripts:
  ```
  cd codes_tps
  bash generate_tps_demo_paired.sh
  bash generate_tps_demo_unpaired.sh
  bash generate_tps_test_paired.sh
  bash generate_tps_test_unpaired.sh
  ```
  The generated tps conditions will be saved under the directory `KGI/example/generate_tps/` and also could be downloaded from [test_tps_conditions](https://drive.google.com/file/d/1JGYukLDJOXRkLmCrddzCJJbcIQhTVFIp/view?usp=drive_link). The files should be placed under the directory `KGI/data/zalando-hd-resized/test/` for final results generation with semantic conditioned inpainting.
* With the trained Semantic Conditioned Inpainting Model and tps conditions, the final results could be generated with the following scripts:
  ```
  cd codes_sci
  bash generate_sci_demo_paired.sh
  bash generate_sci_demo_unpaired.sh
  bash generate_sci_test_paired.sh
  bash generate_sci_test_unpaired.sh
  ```
## Demo with Pretrained Model
   With the pretrained models, the final try-on results and the visualizations of the intermediate results could be generated with the following demo scripts:
   ```
   python3 generate_demo.py
   ```
   The final try-on results will be saved under `KGI/example/generate_demo/final_results/` and the visualizations of the intermediate results will be saved under `KGI/example/generate_demo/vis/`. Below is an example of demo results.
   ![Demo Image 3](https://github.com/lizhi-ntu/KGI/blob/main/imgs/demo_img3.png)
## Acknowledgement and Citations
* The implementation of Keypoints Generator is based on codes repo [SemGCN](https://github.com/garyzhao/SemGCN).
* The implementation of Semantic Conditioned Inpainting Model is based on [semantic-diffusion-model](https://github.com/WeilunWang/semantic-diffusion-model) and [RePaint](https://github.com/andreas128/RePaint).
* The implementation of datasets and dataloader is based on codes repo [HR-VITON](https://github.com/sangyun884/HR-VITON).
* If you find our work is useful, please use the following citation:
  ```
  @InProceedings{Li_2023_ICCV,
    author    = {Li, Zhi and Wei, Pengfei and Yin, Xiang and Ma, Zejun and Kot, Alex C.},
    title     = {Virtual Try-On with Pose-Garment Keypoints Guided Inpainting},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {22788-22797}
  }
  ```
