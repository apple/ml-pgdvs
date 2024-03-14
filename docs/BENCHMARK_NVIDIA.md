# Benchmark: NVIDIA Dynamic Scenes

## Table of Contents

- [1 Download Data](#1-download-data)
- [2 Preprocess](#2-preprocess)
  - [2.1 Use Precomputed Flow and Mask](#21-use-precomputed-flow-and-mask)
  - [2.2 Reproduce Preprocessed Results](#22-reproduce-preprocessed-results)
  - [2.3 [Optional] Compute ZoeDepth](#23-optional-compute-zoedepth)
- [3 Run Benchmark](#3-run-benchmark)
- [4 Run Spatial Temporal Interpolation Visualizations](#3-run-spatial-temporal-interpolation-visualizations)

## 1 Download Data

```bash
# this environment variable is used for demonstration
cd /path/to/this/repo
export PGDVS_ROOT=$PWD
```

For a fair comparison to scene-specific approaches, we evaluate our proposed pipeline on the [DynIBaR](https://github.com/google/dynibar)'s [processed data](https://drive.google.com/drive/folders/1Gv6j_RvDG2WrpqEJWtx73u1tlCZKsPiM):

```bash
# download processed data
conda activate pgdvs
gdown https://drive.google.com/drive/folders/1Gv6j_RvDG2WrpqEJWtx73u1tlCZKsPiM -O ${PGDVS_ROOT}/data/nvidia_long --folder

# unzip
ALL_SCENE_IDS=(Balloon1  Balloon2  Jumping  Playground  Skating  Truck  Umbrella  dynamicFace)
# rgb, poses
printf "%s\0" "${ALL_SCENE_IDS[@]}" | xargs -0 -n 1 -I {} -P 8 unzip ${PGDVS_ROOT}/data/nvidia_long/{}.zip -d ${PGDVS_ROOT}/data/nvidia_long/
# depth
printf "%s\0" "${ALL_SCENE_IDS[@]}" | xargs -0 -n 1 -I {} -P 8 unzip ${PGDVS_ROOT}/data/nvidia_long/Depths/{}_disp.zip -d ${PGDVS_ROOT}/data/nvidia_long/Depths/{}/
```

After running the above command, you should have a structure as the following:
```
.
+-- data
|  +-- nvidia_long
|  |  +-- Balloon1
|  |  +-- Balloon2
|  |  ...
|  |  +-- Depths
|  |  |  +-- Balloon1
|  |  |  |  +-- disp
|  |  |  +-- Balloon2
|  |  |  |  +-- disp
|  |  |  ...
```

## 2 Preprocess

### 2.1 Use Precomputed Flow and Mask

```bash
gdown 1wn9rgCRDWqOJHZmViFYP5vHUbGk81fVp -O ${PGDVS_ROOT}/data/
unzip ${PGDVS_ROOT}/data/nvidia_long_flow_mask.zip -d ${PGDVS_ROOT}/data
```

### 2.2 Reproduce Preprocessed Results

Our pseudo-generalized approach requires optical flow and mask for potentially dynamic content. For this, we need several third parties's repositories and checkpoints. **NOTE**: the `CUDA_HOME` must be set correctly for [`detectron2`](https://github.com/facebookresearch/detectron2)'s installation and consequentially [`OneFormer`](https://github.com/SHI-Labs/OneFormer)'s usage.
```bash
CUDA_HOME=/usr/local/cuda  # set to your own CUDA_HOME, where nvcc is installed
bash ${PGDVS_ROOT}/scripts/preprocess/preprocess.sh \
  ${CUDA_HOME} \
  ${PGDVS_ROOT} \
  ${PGDVS_ROOT}/data \
  prepare
```
After running the command, repositories and pretrained checkpoints will be saved to `${PGDVS_ROOT}/third_parties` and `${PGDVS_ROOT}/ckpts` respectively.

We then compute optical flow and mask with
```bash
CUDA_HOME=/usr/local/cuda  # set to your own CUDA_HOME
bash ${PGDVS_ROOT}/scripts/preprocess/preprocess.sh \
  ${CUDA_HOME} \
  ${PGDVS_ROOT} \
  ${PGDVS_ROOT}/data/nvidia_long \
  execute_on_nvidia \
  ${PGDVS_ROOT}/data/nvidia_long_flow_mask \
  Balloon1  # can be one of [Balloon1  Balloon2  Jumping  Playground  Skating  Truck  Umbrella  dynamicFace]
```
The computed optical flows and masks will be saved to `${PGDVS_ROOT}/data/nvidia_long_flow_mask`

### 2.3 \[Optional\] Compute ZoeDepth

If you want to try using [ZoeDepth](https://github.com/isl-org/ZoeDepth), we provide a preprocessed data:
```bash
gdown 1EcOzg8TSIbQtS7hw9-2RNIvq6L59P4RR -O ${PGDVS_ROOT}/data/
```

Or you can compute them yourself. **Note**, the following command requires that the masks from either [2.1 Use Precomputed Flow and Mask](#21-use-precomputed-flow-and-mask) or [2.2 Reproduce Preprocessed Results](#22-reproduce-preprocessed-results) to align static area to mitigate scale and shift inconsistencies:
```bash
bash ${PGDVS_ROOT}/scripts/preprocess/preprocess.sh \
  /usr/local/cuda/ \
  ${PGDVS_ROOT} \
  ${PGDVS_ROOT}/data/nvidia_long \
  execute_on_nvidia_zoedepth \
  ${PGDVS_ROOT}/data/nvidia_long_zoedepth \
  Balloon1  # can be one of [Balloon1  Balloon2  Jumping  Playground  Skating  Truck  Umbrella  dynamicFace]
```

The computed depths will be saved to `${PGDVS_ROOT}/data/nvidia_long_zoedepth`.

## 3 Run Benchmark

To obtain quantitative results, run the following command

```bash
benchmark_type=default
scene_id='[Balloon1]'  # or 'null' to evaluate on all scenes

bash ${PGDVS_ROOT}/scripts/benchmark.sh \
  ${PGDVS_ROOT} \
  ${PGDVS_ROOT}/ckpts \
  ${PGDVS_ROOT}/data \
  nvidia \
  ${scene_id} \
  ${benchmark_type}
```
You can choose `benchmark_type` from one of the following:

| benchmark_type |  static rendering | dynamic rendering |
|:----------|:-------------| :-------------|
| st_cvd_dy_cvd | **point renderer** from **consistent** depth |  softsplat from **consistent** depth |
| st_cvd_dy_cvd_pcl_clean | **point renderer** from **consistent** depth | softsplat from **consistent** depth and outlier removal for point cloud |
| st_cvd_pcl_clean_dy_cvd_pcl_clean | **point renderer** from **consistent** depth and outlier removal for point cloud | softsplat from **consistent** depth and outlier removal for point cloud |
| st_gnt | GNT with **full** input |  none |
| st_gnt_masked_attn | GNT with **full** input and **masked** attention | none |
| st_gnt_dy_cvd |  GNT with **full** input | softsplat from **consistent** depth |
| st_gnt_dy_cvd_pcl_clean |  GNT with **full** input | softsplat from **consistent** depth and outlier removal for point cloud |
| st_gnt_masked_input_dy_cvd | GNT with **masked** input | softsplat from **consistent** depth |
| st_gnt_masked_input_dy_cvd_pcl_clean | GNT with **masked** input | softsplat from **consistent** depth and outlier removal for point cloud |
| st_gnt_masked_input_attn_dy_cvd_pcl_clean | GNT with **masked** input and **masked** attention | softsplat from **consistent** depth and outlier removal for point cloud |
| st_gnt_masked_attn_dy_cvd_pcl_clean **(default)**  | GNT with **full** input and **masked** attention | softsplat from **consistent** depth and outlier removal for point cloud |
| st_gnt_masked_attn_dy_cvd_pcl_clean_render_point | GNT with **full** input and **masked** attention | **point renderer** from **consistent** depth and outlier removal for point cloud |
| st_gnt_masked_attn_dy_cvd_pcl_clean_render_mesh | GNT with **full** input and **masked** attention | **mesh renderer** from **consistent** depth and outlier removal for point cloud |
| st_gnt_masked_attn_dy_zoed_pcl_clean | GNT with **full** input and **masked** attention | softsplat from **ZoeDepth**, outlier removal for point cloud |
| st_gnt_masked_attn_dy_cvd_pcl_clean_track_tapir | GNT with **full** input and **masked** attention | softsplat from **consistent** depth, outlier removal for point cloud, and **tracking** with TAPIR |
| st_gnt_masked_attn_dy_cvd_pcl_clean_track_cotracker | GNT with **full** input and **masked** attention | from **consistent** depth, outlier removal for point cloud, and **tracking** with CoTracker |
    
All results will be saved to `${PGDVS_ROOT}/experiments`, where subfolder `infos` contain image-wise quantitative results and subfolder `vis` stores image-wise renderings.

Regarding the number of GPUs `benchmark.sh` uses: it will first check whether environment variable `CUDA_VISIBLE_DEVICES` has been set. If yes, then it just uses the number of GPUs specified by `CUDA_VISIBLE_DEVICES`. Otherwise, it will uses all GPUs on the server.

With 8 A100 GPUs, for the evaluation on the whole dataset of 8 scenes (15840 images as the summation of the number of both training views and test views ) with the resolution about 288x550:
- for each `benchmark_type` **without tracking**, the evaluation takes around 2 days
- for `benchmark_type` **with tracking**, tracking with [TAPIR](https://github.com/deepmind/tapnet) needs around 5 days and tracking with [CoTracker](https://github.com/facebookresearch/co-tracker) needs 10 days due to the costly dense tracking. Therefore, for evaluation on these types, we highly recommend parallelizing the evaluation as largely as possible, e.g., evaluating one scene with 8 GPUs.


## 4 Run Spatial Temporal Interpolation Visualizations

All results will be saved to `${PGDVS_ROOT}/experiments`.

```bash
scene_id='[Balloon1]'

bash ${PGDVS_ROOT}/scripts/visualize.sh \
  ${PGDVS_ROOT} \
  ${PGDVS_ROOT}/ckpts \
  ${PGDVS_ROOT}/data \
  nvidia_vis \
  ${scene_id}
```