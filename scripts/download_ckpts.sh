#!/bin/bash
{

DATA_ROOT="$1"
FLAG_ORIGINAL="$2"

printf '\nDownload checkpoints\n'

printf '\nDATA_ROOT: %s' ${DATA_ROOT}
printf '\nFLAG_ORIGINAL: %s\n\n' ${FLAG_ORIGINAL}

mkdir -p ${DATA_ROOT}

eval "$(conda shell.bash hook)"
conda activate pgdvs

download_start="$(date -u +%s)"

if [ "${FLAG_ORIGINAL}" == "1" ]; then
    # GNT
    if [ ! -f ${DATA_ROOT}/gnt/generalized_model_720000.pth ]; then
        gdown 1AMN0diPeHvf2fw53IO5EE2Qp4os5SkoX -O ${DATA_ROOT}/gnt/
    fi

    # TAPIR
    if [ ! -f ${DATA_ROOT}/tapnet/tapir_checkpoint_panning.npy ]; then
        wget https://storage.googleapis.com/dm-tapnet/tapir_checkpoint_panning.npy -P ${DATA_ROOT}/tapnet/
    fi

    # CoTracker
    if [ ! -f ${DATA_ROOT}/cotracker/cotracker_stride_4_wind_8.pth ]; then
        wget https://dl.fbaipublicfiles.com/cotracker/cotracker_stride_4_wind_8.pth -P ${DATA_ROOT}/cotracker/
        wget https://dl.fbaipublicfiles.com/cotracker/cotracker_stride_4_wind_12.pth -P ${DATA_ROOT}/cotracker/
        wget https://dl.fbaipublicfiles.com/cotracker/cotracker_stride_8_wind_16.pth -P ${DATA_ROOT}/cotracker/
    fi

elif [ "${FLAG_ORIGINAL}" == "0" ]; then
    wget https://github.com/apple/ml-pgdvs/releases/download/v0.1/pgdvs_ckpts.zip -P ${DATA_ROOT}/
    unzip ${DATA_ROOT}/pgdvs_ckpts.zip -d ${DATA_ROOT}/

else

    printf '\nHello\n'

fi

download_end="$(date -u +%s)"
download_elapsed="$(($download_end-$download_start))"
printf "\nDownload time elapsed %f\n" $download_elapsed
printf "\n\n"

exit;
}