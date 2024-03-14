# Run PGDVS on an in-the-wild Video

## Table of Contents

- [1 Download Data](#1-download-data)
- [2 Prepare Dependencies](#2-prepare-dependencies)
- [3 Preprocess](#3-preprocess)
  - [3.1 Two-step Camera Pose and Depth Estimations](#31-two-step-camera-pose-and-depth-estimations)
  - [3.2 One-step Camera Pose and Depth estimations](#32-one-step-camera-pose-and-depth-estimations)
- [4 Run Spatial Temporal Interpolation Visualization](#4-run-spatial-temporal-interpolation-visualization)

## 1 Download Data

```bash
# this environment variable is used for demonstration
cd /path/to/this/repo
export PGDVS_ROOT=$PWD
```

We use [DAVIS](https://davischallenge.org/) as an example to illustrate how to render novel view from monocular videos in the wild. First we download the dataset:
```bash
wget https://graphics.ethz.ch/Downloads/Data/Davis/DAVIS-data.zip -P ${PGDVS_ROOT}/data
unzip ${PGDVS_ROOT}/data/DAVIS-data.zip -d ${PGDVS_ROOT}/data
```

## 2 Prepare Dependencies

We need several third parties's repositories and checkpoints. **NOTE**: the `CUDA_HOME` must be set correctly for [`detectron2`](https://github.com/facebookresearch/detectron2)'s installation and consequentially [`OneFormer`](https://github.com/SHI-Labs/OneFormer)'s usage.
```bash
CUDA_HOME=/usr/local/cuda  # set to your own CUDA_HOME, where nvcc is installed
bash ${PGDVS_ROOT}/scripts/preprocess/preprocess.sh \
  ${CUDA_HOME} \
  ${PGDVS_ROOT} \
  ${PGDVS_ROOT}/data \
  prepare
```
After running the command, repositories and pretrained checkpoints will be saved to `${PGDVS_ROOT}/third_parties` and `${PGDVS_ROOT}/ckpts` respectively.

## 3 Preprocess

For a monocular video, we provide two ways to preprocess, i.e., obtaining camera poses, consistent depths, optical flows, and potentially dynamic masks: 
1. **Two-step** camera pose and depth estimations: we need [`COLMAP`](https://github.com/colmap/colmap) for this. We first run `COLMAP` and then apply [Consistent Depth of Moving Objects in Video](https://arxiv.org/abs/2108.01166) (official code is [here](https://github.com/google/dynamic-video-depth) and our modified version is [here](https://github.com/Xiaoming-Zhao/dynamic-video-depth)).
2. **One-step** camera pose and depth estimations: we directly run [Structure and Motion from Casual Videos](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136930020.pdf) (official code is [here](https://github.com/ztzhang/casualSAM) and our modified version is [here](https://github.com/Xiaoming-Zhao/casualSAM)).

**Note 1**: we modify the two consistent depth estimation tools above to tailor to our needs. For the preprocessing, please use **our forked/modified versions** ([1](https://github.com/Xiaoming-Zhao/dynamic-video-depth), [2](https://github.com/Xiaoming-Zhao/casualSAM)), which will be automatically handled by our [preprocess.sh](../scripts/preprocess/preprocess.sh).

**Note 2**: we empirically find that the Two-Step version works better.

### 3.1 Two-step Camera Pose and Depth Estimations

We need [`COLMAP`](https://github.com/colmap/colmap) for this. If `COLMAP` has not been installed yet, you can refer to [install_colmap.sh](../scripts/preprocess/install_colmap.sh) on how to manually install it. You may need to first set an environment variable `NOCONDA_PATH` by putting `export NOCONDA_PATH=$PATH` in your `.bashrc` (or equivalent shell setup file) before `conda` changes `PATH` (see [this issue](https://github.com/pism/pism/issues/356)).

```bash
# Though the script should be able to run all steps automatically,
# for debugging purpose, we would recommend running those commands one by one.
# Namely, you could run each command by commenting out the rest.

CUDA_HOME=/usr/local/cuda  # set to your own CUDA_HOME
SCENE_ID=dog
bash ${PGDVS_ROOT}/scripts/preprocess/preprocess.sh \
  ${CUDA_HOME} \
  ${PGDVS_ROOT} \
  ${PGDVS_ROOT}/data/DAVIS/JPEGImages/480p \
  execute_on_mono_two_step_pose_depth \
  ${PGDVS_ROOT}/data/DAVIS_processed_two_step_pose_depth  \
  ${SCENE_ID} \
  /usr/bin/colmap  # set to your own COLMAP binary file path
```

### 3.2 One-step Camera Pose and Depth Estimations

```bash
# Though the script should be able to run all steps automatically,
# for debugging purpose, we would recommend running those commands one by one.
# Namely, you could run each command by commenting out the rest.

CUDA_HOME=/usr/local/cuda  # set to your own CUDA_HOME
SCENE_ID=dog
bash ${PGDVS_ROOT}/scripts/preprocess/preprocess.sh \
  ${CUDA_HOME} \
  ${PGDVS_ROOT} \
  ${PGDVS_ROOT}/data/DAVIS/JPEGImages/480p \
  execute_on_mono_one_step_pose_depth \
  ${PGDVS_ROOT}/data/DAVIS_processed_one_step_pose_depth \
  ${SCENE_ID}
```

## 4 Run Spatial Temporal Interpolation Visualization

After completing preprocessing, we can run spatial temporal interpolation. Here we use Two-step Camera Pose and Depth Estimations's saved path as an exmaple. The result will be saved to `${PGDVS_ROOT}/experiments`.
```bash
# vis_bt_max_disp:
# - boat: 48
# - dog: 48
# - stroller: 96
# - train: 48

SCENE_ID='[dog]'

bash ${PGDVS_ROOT}/scripts/visualize.sh \
  ${PGDVS_ROOT} \
  ${PGDVS_ROOT}/ckpts \
  ${PGDVS_ROOT}/data/DAVIS_processed_two_step_pose_depth/ \
  mono_vis \
  ${SCENE_ID} \
  engine.engine_cfg.render_cfg.render_stride=2 \
  vis_specifics.vis_center_time=40 \
  vis_specifics.vis_time_interval=30 \
  vis_specifics.vis_bt_max_disp=48 \
  vis_specifics.n_render_frames=100
```