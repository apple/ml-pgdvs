# Benchmark: DyCheck iPhone Dataset

## Table of Contents

- [1 Download Data](#1-download-data)
- [2 Preprocess](#2-preprocess)
  - [2.1 Use Precomputed Flow and Mask](#21-use-precomputed-flow-and-mask)
  - [2.2 Reproduce Preprocessed Results](#22-reproduce-preprocessed-results)
- [3 Run Benchmark](#3-run-benchmark)


## 1 Download Data

```bash
# this environment variable is used for demonstration
cd /path/to/this/repo
export PGDVS_ROOT=$PWD
```

Please follow [DyCheck's official tutorial](https://github.com/KAIR-BAIR/dycheck/blob/ddf77a4e006fdbc5aed28e0859c216da0de5aff5/docs/DATASETS.md#2-iphone-dataset) to download the iPhone dataset to `${PGDVS_ROOT}/data`. After this, you should have a structure as
```
.
+-- data
|  +-- iphone
|  |  +-- apple
|  |  +-- block
|  |  ...
```

## 2 Preprocess

### 2.1 Use Precomputed Flow and Mask

```bash
gdown 1SgvqDJcuFaGJr6Lr3bE9B-knjbOnADQs -O ${PGDVS_ROOT}/data/
unzip ${PGDVS_ROOT}/data/dycheck_iphone_flow_mask.zip -d ${PGDVS_ROOT}/data
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
cd ${PGDVS_ROOT}

bash ${PGDVS_ROOT}/scripts/preprocess/preprocess.sh \
  /usr/local/cuda/ \
  ${PGDVS_ROOT} \
  ${PGDVS_ROOT}/data/iphone \
  execute_on_dycheck \
  ${PGDVS_ROOT}/data/dycheck_iphone_flow_mask \
  apple  # can be one of [apple block paper-windmill space-out spin teddy wheel]
```
The computed optical flows and masks will be saved to `${PGDVS_ROOT}/data/dycheck_iphone_flow_mask`

## 3 Run Benchmark

To obtain quantitative results, run the following command. All results will be saved to `${PGDVS_ROOT}/experiments`.

```bash
benchmark_type=st_gnt_masked_attn_dy_cvd_pcl_clean_render_point
scene_id='[apple]'  # or 'null' to evaluate on all scenes

bash ${PGDVS_ROOT}/scripts/benchmark.sh \
  ${PGDVS_ROOT} \
  ${PGDVS_ROOT}/ckpts \
  ${PGDVS_ROOT}/data \
  dycheck_iphone \
  ${scene_id} \
  ${benchmark_type}
```