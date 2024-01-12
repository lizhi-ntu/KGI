sudo apt install zip unzip libgl1-mesa-glx tmux -y && \
conda install pytorch==1.10.0 torchvision=0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge -y && \
conda install mpi4py -y && \
pip install opencv-python torchgeometry Pillow tqdm tensorboardX scikit-image scipy ipython lpips torchmetrics[image] pyyaml blobfile pyclipper
