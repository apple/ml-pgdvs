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
    RUN_TYPE=(${ALL_ARGS[5]})

    N_IN_ARGS=6

    printf '\nREPO_ROOT: %s' ${REPO_ROOT}
    printf '\nCKPT_ROOT: %s' ${CKPT_ROOT}
    printf '\nDATA_ROOT: %s' ${DATA_ROOT}
    printf '\nDATASET: %s' ${DATASET}
    printf '\nSCENE_ID: %s' ${SCENE_ID}
    printf '\nRUN_TYPE: %s' ${RUN_TYPE}

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

    # for jax
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export TF_CPP_MIN_LOG_LEVEL=0

    VALID_DATA=(nvidia dycheck_iphone)

    if [[ ${VALID_DATA[@]} =~ ${DATASET} ]]
    then
        printf "\n\ndataset ${DATASET} is valid\n"
    else
        printf "\n\ndataset ${DATASET} is NOT supported\n"
        exit;
    fi

    OVERWRITE_CONF=()
    
    if [ "${RUN_TYPE}" == "st_cvd_dy_cvd" ]; then
       
        printf "\n\nRun benchmark ${RUN_TYPE}\n"

        OVERWRITE_CONF+=('static_renderer=geo')
        
        OVERWRITE_CONF+=('engine.engine_cfg.render_cfg.st_pcl_remove_outlier=false')
        
        OVERWRITE_CONF+=('engine.engine_cfg.render_cfg.dyn_pcl_remove_outlier=false')

        OVERWRITE_CONF+=('engine.engine_cfg.render_cfg.st_render_pcl_pt_radius=0.01')
        OVERWRITE_CONF+=('engine.engine_cfg.render_cfg.st_render_pcl_pts_per_pixel=3')

        OVERWRITE_CONF+=('dataset.dataset_list.train=[nvidia_eval_pure_geo]')
        OVERWRITE_CONF+=('dataset.dataset_list.eval=[nvidia_eval_pure_geo]')
        OVERWRITE_CONF+=("dataset.dataset_specifics.nvidia_eval_pure_geo.scene_ids=${SCENE_ID}")
    
    elif [ "${RUN_TYPE}" == "st_cvd_dy_cvd_pcl_clean" ]; then
       
        printf "\n\nRun benchmark ${RUN_TYPE}\n"

        OVERWRITE_CONF+=('static_renderer=geo')

        OVERWRITE_CONF+=('engine.engine_cfg.render_cfg.st_pcl_remove_outlier=false')

        OVERWRITE_CONF+=('engine.engine_cfg.render_cfg.dyn_pcl_remove_outlier=true')

        OVERWRITE_CONF+=('engine.engine_cfg.render_cfg.st_render_pcl_pt_radius=0.01')
        OVERWRITE_CONF+=('engine.engine_cfg.render_cfg.st_render_pcl_pts_per_pixel=3')

        OVERWRITE_CONF+=('dataset.dataset_list.train=[nvidia_eval_pure_geo]')
        OVERWRITE_CONF+=('dataset.dataset_list.eval=[nvidia_eval_pure_geo]')
        OVERWRITE_CONF+=("dataset.dataset_specifics.nvidia_eval_pure_geo.scene_ids=${SCENE_ID}")
    
    elif [ "${RUN_TYPE}" == "st_cvd_pcl_clean_dy_cvd_pcl_clean" ]; then
       
        printf "\n\nRun benchmark ${RUN_TYPE}\n"

        OVERWRITE_CONF+=('static_renderer=geo')

        OVERWRITE_CONF+=('engine.engine_cfg.render_cfg.st_pcl_remove_outlier=true')
        OVERWRITE_CONF+=('engine.engine_cfg.render_cfg.st_pcl_outlier_knn=50')
        OVERWRITE_CONF+=('engine.engine_cfg.render_cfg.st_pcl_outlier_std_thres=0.2')

        OVERWRITE_CONF+=('engine.engine_cfg.render_cfg.dyn_pcl_remove_outlier=true')

        OVERWRITE_CONF+=('engine.engine_cfg.render_cfg.st_render_pcl_pt_radius=0.01')
        OVERWRITE_CONF+=('engine.engine_cfg.render_cfg.st_render_pcl_pts_per_pixel=3')

        OVERWRITE_CONF+=('dataset.dataset_list.train=[nvidia_eval_pure_geo]')
        OVERWRITE_CONF+=('dataset.dataset_list.eval=[nvidia_eval_pure_geo]')
        OVERWRITE_CONF+=("dataset.dataset_specifics.nvidia_eval_pure_geo.scene_ids=${SCENE_ID}")
    
    elif [ "${RUN_TYPE}" == "st_gnt" ]; then
       
        printf "\n\nRun benchmark ${RUN_TYPE}\n"
 
        OVERWRITE_CONF+=('engine.engine_cfg.render_cfg.pure_gnt=true')
        OVERWRITE_CONF+=('engine.engine_cfg.render_cfg.gnt_use_dyn_mask=false')
        OVERWRITE_CONF+=('engine.engine_cfg.render_cfg.gnt_use_masked_spatial_src=false')
   
    elif [ "${RUN_TYPE}" == "st_gnt_masked_attn" ]; then
       
        printf "\n\nRun benchmark ${RUN_TYPE}\n"
 
        OVERWRITE_CONF+=('engine.engine_cfg.render_cfg.pure_gnt_with_dyn_mask=true')
        OVERWRITE_CONF+=('engine.engine_cfg.render_cfg.gnt_use_dyn_mask=true')
        OVERWRITE_CONF+=('engine.engine_cfg.render_cfg.gnt_use_masked_spatial_src=false')
    
    elif [ "${RUN_TYPE}" == "st_gnt_dy_cvd" ]; then
       
        printf "\n\nRun benchmark ${RUN_TYPE}\n"
 
        OVERWRITE_CONF+=('engine.engine_cfg.render_cfg.gnt_use_dyn_mask=false')
        OVERWRITE_CONF+=('engine.engine_cfg.render_cfg.gnt_use_masked_spatial_src=false')
        OVERWRITE_CONF+=('engine.engine_cfg.render_cfg.dyn_pcl_remove_outlier=false')
    
    elif [ "${RUN_TYPE}" == "st_gnt_dy_cvd_pcl_clean" ]; then
       
        printf "\n\nRun benchmark ${RUN_TYPE}\n"
 
        OVERWRITE_CONF+=('engine.engine_cfg.render_cfg.gnt_use_dyn_mask=false')
        OVERWRITE_CONF+=('engine.engine_cfg.render_cfg.gnt_use_masked_spatial_src=false')
        OVERWRITE_CONF+=('engine.engine_cfg.render_cfg.dyn_pcl_remove_outlier=true')
   
    elif [ "${RUN_TYPE}" == "st_gnt_masked_input_dy_cvd" ]; then
       
        printf "\n\nRun benchmark ${RUN_TYPE}\n"
 
        OVERWRITE_CONF+=('engine.engine_cfg.render_cfg.gnt_use_dyn_mask=false')
        OVERWRITE_CONF+=('engine.engine_cfg.render_cfg.gnt_use_masked_spatial_src=true')
        OVERWRITE_CONF+=('engine.engine_cfg.render_cfg.dyn_pcl_remove_outlier=false')
   
    elif [ "${RUN_TYPE}" == "st_gnt_masked_input_attn_dy_cvd_pcl_clean" ]; then
       
        printf "\n\nRun benchmark ${RUN_TYPE}\n"
 
        OVERWRITE_CONF+=('engine.engine_cfg.render_cfg.gnt_use_dyn_mask=true')
        OVERWRITE_CONF+=('engine.engine_cfg.render_cfg.gnt_use_masked_spatial_src=true')
        OVERWRITE_CONF+=('engine.engine_cfg.render_cfg.dyn_pcl_remove_outlier=true')
   
    elif [ "${RUN_TYPE}" == "st_gnt_masked_input_dy_cvd_pcl_clean" ]; then
       
        printf "\n\nRun benchmark ${RUN_TYPE}\n"
 
        OVERWRITE_CONF+=('engine.engine_cfg.render_cfg.gnt_use_dyn_mask=false')
        OVERWRITE_CONF+=('engine.engine_cfg.render_cfg.gnt_use_masked_spatial_src=true')
        OVERWRITE_CONF+=('engine.engine_cfg.render_cfg.dyn_pcl_remove_outlier=true')

    elif [ "${RUN_TYPE}" == "default" ] || [ "${RUN_TYPE}" == "st_gnt_masked_attn_dy_cvd_pcl_clean" ]; then
       
        printf "\n\nRun benchmark ${RUN_TYPE}\n"
   
    elif [ "${RUN_TYPE}" == "st_gnt_masked_attn_dy_cvd_pcl_clean_render_point" ]; then
       
        printf "\n\nRun benchmark ${RUN_TYPE}\n"
 
        OVERWRITE_CONF+=('engine.engine_cfg.render_cfg.gnt_use_dyn_mask=true')
        OVERWRITE_CONF+=('engine.engine_cfg.render_cfg.gnt_use_masked_spatial_src=false')
        OVERWRITE_CONF+=('engine.engine_cfg.render_cfg.dyn_pcl_remove_outlier=true')
        OVERWRITE_CONF+=('engine.engine_cfg.render_cfg.dyn_render_type=pcl')
        OVERWRITE_CONF+=('engine.engine_cfg.render_cfg.dyn_render_pcl_pt_radius=0.01')
        OVERWRITE_CONF+=('engine.engine_cfg.render_cfg.dyn_render_pcl_pts_per_pixel=3')
   
    elif [ "${RUN_TYPE}" == "st_gnt_masked_attn_dy_cvd_pcl_clean_render_mesh" ]; then
       
        printf "\n\nRun benchmark ${RUN_TYPE}\n"
 
        OVERWRITE_CONF+=('engine.engine_cfg.render_cfg.gnt_use_dyn_mask=true')
        OVERWRITE_CONF+=('engine.engine_cfg.render_cfg.gnt_use_masked_spatial_src=false')
        OVERWRITE_CONF+=('engine.engine_cfg.render_cfg.dyn_pcl_remove_outlier=true')
        OVERWRITE_CONF+=('engine.engine_cfg.render_cfg.dyn_render_type=mesh')
   
    elif [ "${RUN_TYPE}" == "st_gnt_masked_attn_dy_zoed_pcl_clean" ]; then
       
        printf "\n\nRun benchmark ${RUN_TYPE}\n"

        if [ "${DATASET}" == "nvidia" ]; then

            OVERWRITE_CONF+=('dataset.dataset_specifics.nvidia_eval.zoe_depth_data_path=nvidia_long_zoedepth.zip')
            OVERWRITE_CONF+=('dataset.dataset_specifics.nvidia_eval.use_zoe_depth=k_me_med_share')
         
        else

            printf "\nDoes not support ${RUN_TYPE} for dataset ${DATASET}\n"

        fi
    
    elif [ "${RUN_TYPE}" == "st_gnt_masked_attn_dy_cvd_pcl_clean_track_tapir" ]; then

        printf "\n\nRun benchmark ${RUN_TYPE}\n"
 
        OVERWRITE_CONF+=('tracker=tapnet')
        OVERWRITE_CONF+=("tracker.ckpt_path=${CKPT_ROOT}/tapnet/tapir_checkpoint_panning.npy")
        OVERWRITE_CONF+=('tracker.query_chunk_size=4096')
        OVERWRITE_CONF+=('tracker.flag_keep_raw_res=false')
        OVERWRITE_CONF+=('engine.engine_cfg.render_cfg.dyn_render_track_temporal=no_tgt')
        OVERWRITE_CONF+=('engine.engine_cfg.render_cfg.dyn_pcl_track_track2base_thres_mult=50')
        OVERWRITE_CONF+=('n_src_views_temporal_track_one_side=5')
     
    elif [ "${RUN_TYPE}" == "st_gnt_masked_attn_dy_cvd_pcl_clean_track_tapir_raw_res" ]; then

        printf "\n\nRun benchmark ${RUN_TYPE}\n"
 
        OVERWRITE_CONF+=('tracker=tapnet')
        OVERWRITE_CONF+=("tracker.ckpt_path=${CKPT_ROOT}/tapnet/tapir_checkpoint_panning.npy")
        OVERWRITE_CONF+=('tracker.query_chunk_size=4096')
        OVERWRITE_CONF+=('tracker.flag_keep_raw_res=true')
        OVERWRITE_CONF+=('engine.engine_cfg.render_cfg.dyn_render_track_temporal=no_tgt')
        OVERWRITE_CONF+=('engine.engine_cfg.render_cfg.dyn_pcl_track_track2base_thres_mult=50')
        OVERWRITE_CONF+=('n_src_views_temporal_track_one_side=5')
    
    elif [ "${RUN_TYPE}" == "st_gnt_masked_attn_dy_cvd_pcl_clean_track_cotracker" ]; then

        printf "\n\nRun benchmark ${RUN_TYPE}\n"
 
        OVERWRITE_CONF+=('tracker=cotracker')
        OVERWRITE_CONF+=("tracker.ckpt_path=${CKPT_ROOT}/cotracker/cotracker_stride_4_wind_8.pth")
        OVERWRITE_CONF+=('tracker.query_chunk_size=1024')
        OVERWRITE_CONF+=('engine.engine_cfg.render_cfg.dyn_render_track_temporal=no_tgt')
        OVERWRITE_CONF+=('engine.engine_cfg.render_cfg.dyn_pcl_track_track2base_thres_mult=50')
        OVERWRITE_CONF+=('n_src_views_temporal_track_one_side=5')
    
    elif [ "${RUN_TYPE}" == "visualize_nvidia_max_disp_32" ]; then

        printf "\n\nRun visualizations\n"
 
        OVERWRITE_CONF+=('resume=vis_wo_resume')
        OVERWRITE_CONF+=('engine=visualizer_pgdvs')
        OVERWRITE_CONF+=('dataset.dataset_list.vis=[nvidia_vis]')
        OVERWRITE_CONF+=('dataset.dataset_specifics.nvidia_vis.n_render_frames=400')
        OVERWRITE_CONF+=('dataset.dataset_specifics.nvidia_vis.vis_center_time=50')
        OVERWRITE_CONF+=('dataset.dataset_specifics.nvidia_vis.vis_time_interval=50')
        OVERWRITE_CONF+=('dataset.dataset_specifics.nvidia_vis.vis_bt_max_disp=32')
        OVERWRITE_CONF+=("dataset.dataset_specifics.nvidia_vis.scene_ids=${SCENE_ID}")
    
    elif [ "${RUN_TYPE}" == "visualize_nvidia_max_disp_64" ]; then

        printf "\n\nRun visualizations\n"
 
        OVERWRITE_CONF+=('resume=vis_wo_resume')
        OVERWRITE_CONF+=('engine=visualizer_pgdvs')
        OVERWRITE_CONF+=('dataset.dataset_list.vis=[nvidia_vis]')
        OVERWRITE_CONF+=('dataset.dataset_specifics.nvidia_vis.n_render_frames=400')
        OVERWRITE_CONF+=('dataset.dataset_specifics.nvidia_vis.vis_center_time=50')
        OVERWRITE_CONF+=('dataset.dataset_specifics.nvidia_vis.vis_time_interval=50')
        OVERWRITE_CONF+=('dataset.dataset_specifics.nvidia_vis.vis_bt_max_disp=64')
        OVERWRITE_CONF+=("dataset.dataset_specifics.nvidia_vis.scene_ids=${SCENE_ID}")
    
    else

        printf '\n\nHello\n'

    fi

    if [ "${DATASET}" == "dycheck_iphone" ]; then

        OVERWRITE_CONF+=("engine.engine_cfg.quant_type=dycheck_iphone")
        OVERWRITE_CONF+=('dataset.dataset_list.train=[dycheck_iphone_eval]')
        OVERWRITE_CONF+=('dataset.dataset_list.eval=[dycheck_iphone_eval]')
        OVERWRITE_CONF+=("dataset.dataset_specifics.dycheck_iphone_eval.scene_ids=${SCENE_ID}")

        OVERWRITE_CONF+=("dataset.dataset_specifics.dycheck_iphone_eval.spatial_src_view_type=clustered")
        OVERWRITE_CONF+=("dataset.dataset_specifics.dycheck_iphone_eval.n_src_views_spatial_cluster=40")

    fi

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
        resume="eval_wo_resume" \
        resume_dir=null \
        engine=evaluator_pgdvs \
        engine.engine_cfg.quant_type=nvidia \
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
        dataset=combined \
        'dataset.dataset_list.train=[nvidia_eval]' \
        'dataset.dataset_list.eval=[nvidia_eval]' \
        "dataset.dataset_specifics.nvidia_eval.scene_ids=${SCENE_ID}" \
        dataset.data_root=${DATA_ROOT} \
        n_dataloader_workers=1 \
        dataset_max_hw=-1 \
        dataset.use_aug=false \
        ${OVERWRITE_STR}
    
    exit;
}