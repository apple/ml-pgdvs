#!/bin/bash
{
    CUDA_HOME="$1"
    REPO_ROOT="$2"
    DATA_ROOT="$3"
    RUN_TYPE="$4"
    SAVE_ROOT="$5"
    SCENE_ID="$6"
    COLMAP_BIN="$7"

    printf '\nCUDA_HOME: %s' ${CUDA_HOME}
    printf '\nREPO_ROOT: %s' ${REPO_ROOT}
    printf '\nDATA_ROOT: %s' ${DATA_ROOT}
    printf '\nRUN_TYPE: %s' ${RUN_TYPE}
    printf '\nSAVE_ROOT: %s' ${SAVE_ROOT}
    printf '\nSCENE_ID: %s' ${SCENE_ID}
    printf '\nCOLMAP_BIN: %s\n' ${COLMAP_BIN}

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

    # FLOW_MODEL=flowformer
    # FLOW_CKPT_F=${REPO_ROOT}/ckpts/flowformer/FlowFormer-Models/sintel.pth
    FLOW_MODEL=raft
    FLOW_CKPT_F=${REPO_ROOT}/ckpts/raft/models/raft-things.pth

    if [ "${RUN_TYPE}" == "prepare" ]; then
        
        cd ${REPO_ROOT}
        mkdir -p ${REPO_ROOT}/third_parties

        # for optical flow
        pip install loguru==0.7.0 wandb==0.15.8

        if [ ! -d ${REPO_ROOT}/third_parties/RAFT ]; then
            git clone git@github.com:princeton-vl/RAFT.git ${REPO_ROOT}/third_parties/RAFT
            cd ${REPO_ROOT}/third_parties/RAFT
            git checkout 3fa0bb0a9c633ea0a9bb8a79c576b6785d4e6a02
            cd ${REPO_ROOT}

            if [ ! -f ${REPO_ROOT}/ckpts/raft/models.zip ]; then
                wget https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip -P ${REPO_ROOT}/ckpts/raft/
                unzip ${REPO_ROOT}/ckpts/raft/models.zip -d ${REPO_ROOT}/ckpts/raft/
            fi
        fi

        if [ ! -d ${REPO_ROOT}/third_parties/FlowFormer ]; then
            git clone git@github.com:drinkingcoder/FlowFormer-Official.git ${REPO_ROOT}/third_parties/FlowFormer
            cd ${REPO_ROOT}/third_parties/FlowFormer
            git checkout 6ba7ea82b45394d3a7a25808399695427c9febd8
            git apply ${REPO_ROOT}/third_parties/flowformer_6ba7ea82.patch --verbose
            cd ${REPO_ROOT}

            if [ ! -d ${PGDVS_ROOT}/ckpts/flowformer/ ]; then
                gdown https://drive.google.com/drive/folders/1K2dcWxaqOLiQ3PoqRdokrgWsGIf3yBA_ -O ${PGDVS_ROOT}/ckpts/flowformer/ --folder
            fi
        fi

        # for mask
        pip install ftfy==6.1.1 regex==2023.6.3 diffdist==0.1

        pip install https://shi-labs.com/natten/wheels/cu117/torch2.0.0/natten-0.14.6+torch200cu117-cp310-cp310-linux_x86_64.whl
        
        if [ ! -d ${REPO_ROOT}/third_parties/OneFormer ]; then
            git clone git@github.com:SHI-Labs/OneFormer.git ${REPO_ROOT}/third_parties/OneFormer
            cd ${REPO_ROOT}/third_parties/OneFormer
            git checkout 56799ef9e02968af4c7793b30deabcbeec29ffc0

            # compilation for OneFormer
            cd ${REPO_ROOT}/third_parties/OneFormer/oneformer/modeling/pixel_decoder/ops
            bash make.sh
            cd ${REPO_ROOT}

            if [ ! -f ${REPO_ROOT}/ckpts/oneformer/250_16_dinat_l_oneformer_ade20k_160k.pth ]; then
                wget https://shi-labs.com/projects/oneformer/ade20k/250_16_dinat_l_oneformer_ade20k_160k.pth -P ${REPO_ROOT}/ckpts/oneformer/
                wget https://shi-labs.com/projects/oneformer/coco/150_16_dinat_l_oneformer_coco_100ep.pth -P ${REPO_ROOT}/ckpts/oneformer/
            fi
        fi

        # install detectron2
        # pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
        if [ ! -d ${REPO_ROOT}/third_parties/detectron2 ]; then
            git clone https://github.com/facebookresearch/detectron2 ${REPO_ROOT}/third_parties/detectron2
            cd ${REPO_ROOT}/third_parties/detectron2
            git checkout 57bdb21249d5418c130d54e2ebdc94dda7a4c01a
            CUDA_HOME=${CUDA_HOME} pip install -e .

            cd ${REPO_ROOT}
        fi

        if [ ! -d ${REPO_ROOT}/third_parties/FastSAM ]; then
            pip install ultralytics==8.0.120
            pip install git+https://github.com/openai/CLIP.git@a1d071733d7111c9c014f024669f959182114e33

            git clone git@github.com:CASIA-IVA-Lab/FastSAM.git ${REPO_ROOT}/third_parties/FastSAM
            cd ${REPO_ROOT}/third_parties/FastSAM
            git checkout 2d13729519733be36423884ed361aa2aee41c381
            cd ${REPO_ROOT}

            if [ ! -d ${REPO_ROOT}/ckpts/FastSAM/ ]; then
                gdown 1m1sjY4ihXBU1fZXdQ-Xdj-mDltW-2Rqv -O ${REPO_ROOT}/ckpts/FastSAM/
            fi
        fi

        pip install git+https://github.com/facebookresearch/segment-anything.git@6fdee8f2727f4506cfbbe553e23b895e27956588

        if [ ! -d ${REPO_ROOT}/ckpts/SAM/sam_vit_h_4b8939.pth ]; then
            wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P ${REPO_ROOT}/ckpts/SAM/
        fi

        # For Depth
        if [ ! -d ${REPO_ROOT}/third_parties/ZoeDepth ]; then
            git clone git@github.com:isl-org/ZoeDepth.git ${REPO_ROOT}/third_parties/ZoeDepth
            cd ${REPO_ROOT}/third_parties/ZoeDepth/
            git checkout edb6daf45458569e24f50250ef1ed08c015f17a7
            cd ${REPO_ROOT}

            if [ ! -f ${REPO_ROOT}/ckpts/zoe_depth/ZoeD_M12_NK.pt ]; then
                wget https://github.com/isl-org/ZoeDepth/releases/download/v1.0/ZoeD_M12_N.pt -P ${REPO_ROOT}/ckpts/zoe_depth/
                wget https://github.com/isl-org/ZoeDepth/releases/download/v1.0/ZoeD_M12_K.pt -P ${REPO_ROOT}/ckpts/zoe_depth/
                wget https://github.com/isl-org/ZoeDepth/releases/download/v1.0/ZoeD_M12_NK.pt -P ${REPO_ROOT}/ckpts/zoe_depth/
            fi
        fi

        if [ ! -d ${REPO_ROOT}/third_parties/dynamic_video_depth ]; then
            git clone --recursive git@github.com:Xiaoming-Zhao/dynamic-video-depth.git ${REPO_ROOT}/third_parties/dynamic_video_depth

            if [ ! -f ${REPO_ROOT}/ckpts/MiDaS/dpt_beit_large_512.pt ]; then
                # MiDaS 1645b7e1675301fdfac03640738fe5a6531e17d6
                wget https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_large_512.pt -P ${REPO_ROOT}/ckpts/MiDaS/
                wget https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_384.pt -P ${REPO_ROOT}/ckpts/MiDaS/
            fi
        fi

        pip install configargparse==1.7

        if [ ! -d ${REPO_ROOT}/third_parties/casualSAM ]; then
            git clone --recursive git@github.com:Xiaoming-Zhao/casualSAM.git ${REPO_ROOT}/third_parties/casualSAM

            cd ${REPO_ROOT}/third_parties/casualSAM

            if [ ! -d ${REPO_ROOT}/third_parties/casualSAM/pretrained_depth_ckpt ]; then
                mkdir pretrained_depth_ckpt
                wget https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_384.pt -O ${REPO_ROOT}/third_parties/casualSAM/pretrained_depth_ckpt/midas_cpkt.pt
            fi

            if [ ! -d ${REPO_ROOT}/third_parties/casualSAM/third_party/RAFT/models.zip ]; then
                wget https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip -P ${REPO_ROOT}/third_parties/casualSAM/third_party/RAFT/
                unzip ./third_party/RAFT/models.zip -d ${REPO_ROOT}/third_parties/casualSAM/third_party/RAFT/
            fi

            cd ${REPO_ROOT}
        fi
    
    elif [ "${RUN_TYPE}" == "execute_on_nvidia" ]; then

        cd ${REPO_ROOT}

        # We only need to compute optical flow and masks

        python ${REPO_ROOT}/pgdvs/preprocess/compute_flow.py \
            --model_to_use ${FLOW_MODEL} \
            --ckpt_f ${FLOW_CKPT_F} \
            --root_dir ${DATA_ROOT}/${SCENE_ID}/dense \
            --save_dir ${SAVE_ROOT}/${SCENE_ID}/dense \
            --image_subdir 'images_*x288' \
            --img_pair_max_diff 2  # we need interval=2 because we also render image for frames of the monocular video input

        export IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg && \
        python ${REPO_ROOT}/pgdvs/preprocess/compute_mask.py \
            --root_dir ${SAVE_ROOT}/${SCENE_ID}/dense \
            --save_dir ${SAVE_ROOT}/${SCENE_ID}/dense \
            --image_subdir rgbs \
            --mask_type semantic \
            --oneformer_ckpt_dir ${REPO_ROOT}/ckpts/oneformer \
            --sam_ckpt_f ${REPO_ROOT}/ckpts/SAM/sam_vit_h_4b8939.pth
        
        # clean up
        rm -r ${SAVE_ROOT}/${SCENE_ID}/dense/rgbs
        rm -r ${SAVE_ROOT}/${SCENE_ID}/dense/poses_bounds_cvd
    
    elif [ "${RUN_TYPE}" == "execute_on_nvidia_zoedepth" ]; then

        ALL_TYPES=(K)
        for TMP_TYPE in "${ALL_TYPES[@]}";
        do
            python ${REPO_ROOT}/pgdvs/preprocess/compute_zoedepth.py \
                --root_dir ${DATA_ROOT}/${SCENE_ID}/dense \
                --mask_dir ${DATA_ROOT}/../nvidia_long_flow_mask/${SCENE_ID}/dense \
                --save_dir ${SAVE_ROOT}/${SCENE_ID}/dense \
                --image_subdir 'images_*x288' \
                --zoedepth_type ${TMP_TYPE} \
                --zoedepth_ckpt_dir ${REPO_ROOT}/ckpts/zoe_depth
        done
    
    elif [ "${RUN_TYPE}" == "execute_on_dycheck" ]; then

        cd ${REPO_ROOT}

        MULTI_SEQ=(apple block paper-windmill space-out spin teddy wheel)

        # extract necessary information
        python ${REPO_ROOT}/pgdvs/preprocess/dycheck_mono_info_extractor.py \
            --data_dir ${DATA_ROOT} \
            --save_dir ${SAVE_ROOT} \
            --scene_id ${SCENE_ID}
        
        # compute optical flow on raw images
        python ${REPO_ROOT}/pgdvs/preprocess/compute_flow.py \
            --model_to_use ${FLOW_MODEL} \
            --ckpt_f ${FLOW_CKPT_F} \
            --root_dir ${SAVE_ROOT}/${SCENE_ID} \
            --save_dir ${SAVE_ROOT}/${SCENE_ID} \
            --image_subdir rgbs \
            --image_pattern '*' \
            --img_pair_max_diff 1

        # compute mask for raw images
        export IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg && \
        python ${REPO_ROOT}/pgdvs/preprocess/compute_mask.py \
            --root_dir ${SAVE_ROOT}/${SCENE_ID} \
            --save_dir ${SAVE_ROOT}/${SCENE_ID} \
            --image_subdir rgbs \
            --mask_type semantic \
            --oneformer_ckpt_dir ${REPO_ROOT}/ckpts/oneformer \
            --sam_ckpt_f ${REPO_ROOT}/ckpts/SAM/sam_vit_h_4b8939.pth \
            --flag_dycheck_format
    
    elif [ "${RUN_TYPE}" == "execute_on_mono_one_step_pose_depth" ]; then

        cd ${REPO_ROOT}/third_parties/casualSAM
        
        python train_slam.py \
        --config  ${REPO_ROOT}/third_parties/casualSAM/experiments/davis/default_config.yaml \
        --dataset_name mono \
        --track_name ${SCENE_ID} \
        --mono_data_root ${DATA_ROOT}/${SCENE_ID} \
        --log_dir ${SAVE_ROOT}/${SCENE_ID} \
        --prefix casual_sam \
        --save_disk

        cd ${REPO_ROOT}

        cp -r ${DATA_ROOT}/${SCENE_ID} ${SAVE_ROOT}/${SCENE_ID}/rgbs

        python ${REPO_ROOT}/pgdvs/preprocess/convert_casual_sam_output.py \
            --casual_sam_dir ${SAVE_ROOT}/${SCENE_ID}/casual_sam \
            --rgb_dir ${SAVE_ROOT}/${SCENE_ID}/rgbs \
            --save_dir ${SAVE_ROOT}/${SCENE_ID}

        python ${REPO_ROOT}/pgdvs/preprocess/compute_flow.py \
            --model_to_use ${FLOW_MODEL} \
            --ckpt_f ${FLOW_CKPT_F} \
            --root_dir ${SAVE_ROOT}/${SCENE_ID} \
            --save_dir ${SAVE_ROOT}/${SCENE_ID} \
            --image_subdir null \
            --img_pair_max_diff 1
        
        export IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg && \
        python ${REPO_ROOT}/pgdvs/preprocess/compute_mask.py \
            --root_dir ${SAVE_ROOT}/${SCENE_ID} \
            --save_dir ${SAVE_ROOT}/${SCENE_ID} \
            --image_subdir rgbs \
            --mask_type semantic \
            --oneformer_ckpt_dir ${REPO_ROOT}/ckpts/oneformer \
            --sam_ckpt_f ${REPO_ROOT}/ckpts/SAM/sam_vit_h_4b8939.pth
    
    elif [ "${RUN_TYPE}" == "execute_on_mono_two_step_pose_depth" ]; then
        
        cd ${REPO_ROOT}

        MIDAS_TYPE=dpt_beit_large_512  # use midas_v21_384 if the GPU RAM is OOM

        # compute optical flow on distorted/raw images
        python ${REPO_ROOT}/pgdvs/preprocess/compute_flow.py \
            --model_to_use ${FLOW_MODEL} \
            --ckpt_f ${FLOW_CKPT_F} \
            --root_dir ${DATA_ROOT}/${SCENE_ID} \
            --save_dir ${SAVE_ROOT}/${SCENE_ID}/for_colmap \
            --image_subdir null \
            --img_pair_max_diff 1 \
            --for_colmap

        # compute mask for distorted/raw images
        export IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg && \
        python ${REPO_ROOT}/pgdvs/preprocess/compute_mask.py \
            --root_dir ${SAVE_ROOT}/${SCENE_ID}/for_colmap \
            --save_dir ${SAVE_ROOT}/${SCENE_ID}/for_colmap \
            --image_subdir rgbs_for_colmap \
            --mask_type semantic \
            --oneformer_ckpt_dir ${REPO_ROOT}/ckpts/oneformer \
            --sam_ckpt_f ${REPO_ROOT}/ckpts/SAM/sam_vit_h_4b8939.pth \
            --for_colmap

        # run COLMAP
        python ${REPO_ROOT}/pgdvs/preprocess/colmap_processor.py \
            --cuda_device 0 \
            --colmap_bin ${COLMAP_BIN} \
            --image_path ${SAVE_ROOT}/${SCENE_ID}/for_colmap/rgbs_for_colmap \
            --mask_path ${SAVE_ROOT}/${SCENE_ID}/for_colmap/masks_for_colmap \
            --workspace_path ${SAVE_ROOT}/${SCENE_ID}/colmap \
            --undistort_path ${SAVE_ROOT}/${SCENE_ID}/undistorted \
            --overwrite

        # compute optical flow on undistorted images
        python ${REPO_ROOT}/pgdvs/preprocess/compute_flow.py \
            --model_to_use ${FLOW_MODEL} \
            --ckpt_f ${FLOW_CKPT_F} \
            --root_dir ${SAVE_ROOT}/${SCENE_ID}/undistorted/images \
            --save_dir ${SAVE_ROOT}/${SCENE_ID}/ \
            --image_subdir null \
            --img_pair_max_diff 1

        # compute mask for undistorted images
        export IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg && \
        python ${REPO_ROOT}/pgdvs/preprocess/compute_mask.py \
            --root_dir ${SAVE_ROOT}/${SCENE_ID}/ \
            --save_dir ${SAVE_ROOT}/${SCENE_ID}/ \
            --image_subdir rgbs \
            --mask_type semantic \
            --oneformer_ckpt_dir ${REPO_ROOT}/ckpts/oneformer \
            --sam_ckpt_f ${REPO_ROOT}/ckpts/SAM/sam_vit_h_4b8939.pth

        # convert COLMAP output
        python ${REPO_ROOT}/pgdvs/preprocess/convert_colmap_output.py \
            --data_path ${SAVE_ROOT}/${SCENE_ID}/undistorted \
            --save_dir ${SAVE_ROOT}/${SCENE_ID}
        
        # prepare for consistent depth estimations
        bash ${REPO_ROOT}/third_parties/dynamic_video_depth/scripts/preprocess/mono/run.sh \
            ${REPO_ROOT}/third_parties/dynamic_video_depth \
            ${SAVE_ROOT}/${SCENE_ID} \
            ${SAVE_ROOT}/${SCENE_ID}/for_dynamic_video_depth \
            ${SCENE_ID} \
            ${MIDAS_TYPE} 
        
        # run consistent depth estimations
        cd ${REPO_ROOT}/third_parties/dynamic_video_depth
        bash ${REPO_ROOT}/third_parties/dynamic_video_depth/experiments/mono/train_sequence.sh 0 \
            --mono_data_root ${SAVE_ROOT}/${SCENE_ID}/for_dynamic_video_depth \
            --midas_ckpt_dir ${REPO_ROOT}/third_parties/dynamic_video_depth/checkpoints/midas \
            --logdir ${SAVE_ROOT}/${SCENE_ID}/dynamic_video_depth \
            --tensorboard_keyword none \
            --disp_mul 0.1 \
            --dynibar_depth_mul 10 \
            --gaps 1,2,4,6,8 \
            --lr 1e-6 \
            --scene_lr_mul 1000 \
            --epoch 20 \
            --save_net 10 \
            --follow_dynibar \
            --midas_model_type ${MIDAS_TYPE} \
            --test_output_dir ${SAVE_ROOT}/${SCENE_ID}/dynamic_video_depth/test/

            # --pure_test
        
        cd ${REPO_ROOT}

        # extract depth information
        python ${REPO_ROOT}/pgdvs/preprocess/convert_dyn_video_depth_output.py \
            --base_dir ${SAVE_ROOT}/${SCENE_ID}/dynamic_video_depth \
            --rgb_dir ${SAVE_ROOT}/${SCENE_ID}/rgbs \
            --save_dir ${SAVE_ROOT}/${SCENE_ID}
        
        # clean up: We only need rgbs, depths, poses, masks, flows
        rm -r ${SAVE_ROOT}/${SCENE_ID}/for_colmap/
        rm -r ${SAVE_ROOT}/${SCENE_ID}/colmap/
        rm -r ${SAVE_ROOT}/${SCENE_ID}/undistorted/
        rm -r ${SAVE_ROOT}/${SCENE_ID}/for_dynamic_video_depth/
        rm -r ${SAVE_ROOT}/${SCENE_ID}/dynamic_video_depth/
    
    else

        printf '\nHello\n'

    fi

    exit;
}