#!/bin/bash
{

    ALL_ARGS=("$@")  # treat input as array

    printf '\nAll args: %s' ${ALL_ARGS[@]}

    printf '\n'

    REPO_ROOT=(${ALL_ARGS[0]})
    CKPT_ROOT=(${ALL_ARGS[1]})
    DATA_ROOT=(${ALL_ARGS[2]})
    DATASET=(${ALL_ARGS[3]})
    SCENE_ID=(${ALL_ARGS[4]})
    # RUN_TYPE=(${ALL_ARGS[5]})
    # CUDA_HOME=(${ALL_ARGS[4]})

    N_IN_ARGS=5

    printf '\nREPO_ROOT: %s' ${REPO_ROOT}
    printf '\nCKPT_ROOT: %s' ${CKPT_ROOT}
    printf '\nDATA_ROOT: %s' ${DATA_ROOT}
    printf '\nDATASET: %s' ${DATASET}
    printf '\nSCENE_ID: %s' ${SCENE_ID}
    # printf '\nRUN_TYPE: %s' ${RUN_TYPE}
    # printf '\nCUDA_HOME\n: %s' ${CUDA_HOME}

    eval "$(conda shell.bash hook)"
    conda activate pgdvs

    cd ${REPO_ROOT}
    export PYTHONPATH=${REPO_ROOT}:${PYTHONPATH}

    ulimit -n 65000;
    ulimit -c 0;   # Disable core file creation

    export MKL_THREADING_LAYER=GNU;
    export NCCL_P2P_DISABLE=1;
    export HYDRA_FULL_ERROR=1;
    export OC_CAUSE=1;
    # export CUDA_HOME=${CUDA_HOME}

    # for jax
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export TF_CPP_MIN_LOG_LEVEL=0

    VALID_DATA=(nvidia_vis mono_vis)

    if [[ ${VALID_DATA[@]} =~ ${DATASET} ]]
    then
        printf "\n\ndataset ${DATASET} is valid\n"
    else
        printf "\n\ndataset ${DATASET} is NOT supported\n"
    fi

    OVERWRITE_CONF=()

    if [ "${DATASET}" == "nvidia_vis" ]; then
        
        OVERWRITE_CONF+=('dataset.dataset_list.vis=[nvidia_vis]')
        OVERWRITE_CONF+=("dataset.dataset_specifics.nvidia_vis.scene_ids=${SCENE_ID}")

    elif [ "${DATASET}" == "mono_vis" ]; then
        
        OVERWRITE_CONF+=('dataset.dataset_list.vis=[mono_vis]')
        OVERWRITE_CONF+=("dataset.dataset_specifics.mono_vis.scene_ids=${SCENE_ID}")

    fi

    OVERWRITE_CONF+=('vis_specifics.n_render_frames=400')
    OVERWRITE_CONF+=('vis_specifics.vis_center_time=50')
    OVERWRITE_CONF+=('vis_specifics.vis_time_interval=50')
    OVERWRITE_CONF+=('vis_specifics.vis_bt_max_disp=32')

    OVERWRITE_ARGS=("${ALL_ARGS[@]:${N_IN_ARGS}}")  # remove the first several elements
    OVERWRITE_CONF+=(${OVERWRITE_ARGS[@]})

    OVERWRITE_STR=$(printf " %s" "${OVERWRITE_CONF[@]}")
    OVERWRITE_STR=${OVERWRITE_STR:1}

    printf "\n\nOverwrites: %s" ${OVERWRITE_STR}
    printf "\n"

    # https://stackoverflow.com/a/76742577
    N_GPU_SPLIT_STR=(${CUDA_VISIBLE_DEVICES//,/ })
    NUM_GPUS=${#N_GPU_SPLIT_STR[@]}

    if [ "${NUM_GPUS}" == "0" ]; then
        NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
    fi

    printf "\nNumber of GPU is ${NUM_GPUS}\n\n"

    export IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg && \
    python ${REPO_ROOT}/pgdvs/run.py \
        verbose=true \
        distributed=true \
        seed=0 \
        resume="vis_wo_resume" \
        resume_dir=null \
        engine=visualizer_pgdvs \
        model=pgdvs_renderer \
        model.softsplat_metric_abs_alpha=100.0 \
        static_renderer=gnt \
        static_renderer.model_cfg.ckpt_path=${CKPT_ROOT}/gnt/model_720000.pth \
        series_eval=false \
        eval_batch_size=${NUM_GPUS} \
        n_max_eval_data=-1 \
        eval_save_individual=true \
        engine.engine_cfg.render_cfg.render_stride=1 \
        engine.engine_cfg.render_cfg.chunk_size=2048 \
        engine.engine_cfg.render_cfg.sample_inv_uniform=true \
        engine.engine_cfg.render_cfg.n_coarse_samples_per_ray=256 \
        engine.engine_cfg.render_cfg.n_fine_samples_per_ray=0 \
        engine.engine_cfg.render_cfg.mask_oob_n_proj_thres=1 \
        engine.engine_cfg.render_cfg.mask_invalid_n_proj_thres=4 \
        engine.engine_cfg.render_cfg.dyn_pcl_remove_outlier=true \
        engine.engine_cfg.render_cfg.dyn_pcl_outlier_knn=50 \
        engine.engine_cfg.render_cfg.dyn_pcl_outlier_std_thres=0.1 \
        engine.engine_cfg.render_cfg.gnt_use_dyn_mask=true \
        engine.engine_cfg.render_cfg.gnt_use_masked_spatial_src=false \
        engine.engine_cfg.render_cfg.dyn_render_use_flow_consistency=false \
        dataset=combined \
        'dataset.dataset_list.train=[nvidia_eval]' \
        'dataset.dataset_list.eval=[nvidia_eval]' \
        'dataset.dataset_list.vis=[nvidia_vis]' \
        "dataset.dataset_specifics.mono_vis.scene_ids=${SCENE_ID}" \
        dataset.data_root=${DATA_ROOT} \
        n_dataloader_workers=1 \
        dataset_max_hw=-1 \
        dataset.use_aug=false \
        ${OVERWRITE_STR}
    
    exit;
}