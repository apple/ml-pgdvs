# Modified from: https://github.com/facebookresearch/consistent_depth/blob/e2c9b724d3221aa7c0bf89aa9449ae33b418d943/scripts/install_colmap_ubuntu.sh

export BASE_DIR=$1  # root directory where you prefer to install for colmap and related repositories
echo "Dir: $BASE_DIR"

# IMPORTANT !!!
# See issues here: https://github.com/pism/pism/issues/356
# Make sure that we have a PATH without conda's pollution
# Put the following two lines in .bashrc or .zshrc before conda overwrite the PATH
# export NOCONDA_PATH=$PATH
export CUR_NOCONDA_PATH=$NOCONDA_PATH

# conda deactivate

sudo apt-get update
sudo apt-get install -y libgl1-mesa-glx ffmpeg libsm6 libxext6

# https://colmap.github.io/install.html
sudo apt-get install -y \
    git \
    cmake \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libboost-test-dev \
    libeigen3-dev \
    libsuitesparse-dev \
    libfreeimage-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev

sudo apt-get install -y libcgal-qt5-dev

sudo apt-get install -y libatlas-base-dev libsuitesparse-dev

eval "$(conda shell.bash hook)"
conda activate base
conda install cmake -y

cd $BASE_DIR
git clone https://ceres-solver.googlesource.com/ceres-solver
cd ceres-solver
# git checkout $(git describe --tags) # Checkout the latest release
git checkout 2.1.0
mkdir build
cd build
PATH=$CUR_NOCONDA_PATH cmake .. -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DCMAKE_INSTALL_PREFIX=$BASE_DIR/ceres-solver/build
make -j
make install

# resolve building bugs
sudo apt-get install -y libgoogle-glog-dev
sudo apt-get install -y libfreeimage-dev
sudo apt-get install -y libglu1-mesa-dev freeglut3-dev mesa-common-dev
sudo apt-get install -y libglew-dev
sudo apt-get install -y qt5-default

# https://github.com/facebookresearch/habitat-sim/issues/971#issuecomment-818560795
# https://github.com/colmap/colmap/issues/1271#issuecomment-931900582
# https://github.com/NVIDIA/libglvnd
sudo apt-get install -y lsb-core
sudo apt-get install -y autoconf
sudo apt-get install -y libxext-dev libx11-dev x11proto-gl-dev
cd $BASE_DIR
git clone https://github.com/NVIDIA/libglvnd.git
cd libglvnd
git checkout c8ee005
./autogen.sh
mkdir build
./configure --prefix=$BASE_DIR/libglvnd/build
make -j
make install

# # https://github.com/colmap/colmap/issues/188
# conda activate base
# conda uninstall libtiff

cd $BASE_DIR
git clone https://github.com/colmap/colmap.git
cd colmap
git checkout ea40ef9a  # 3.7
mkdir build
cd build
PATH=$CUR_NOCONDA_PATH cmake .. \
  -DCMAKE_INSTALL_PREFIX=$BASE_DIR/colmap/build \
  -DCMAKE_PREFIX_PATH=$BASE_DIR/ceres-solver/build \
  -DDCMAKE_INCLUDE_PATH=$BASE_DIR/libglvnd/build/include \
  -DCMAKE_LIBRARY_PATH=$BASE_DIR/libglvnd/build/lib
make -j
make install