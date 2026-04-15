export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/datashare/DiffusionDrive/download/maps"
export NAVSIM_EXP_ROOT="/datashare/DiffusionDrive/exp"
export NAVSIM_DEVKIT_ROOT="/data/workspace/shuliu/DiffusionDrive/navsim"
export OPENSCENE_DATA_ROOT="/datashare/DiffusionDrive/download"

export CUDA_VISIBLE_DEVICES=3
export HYDRA_FULL_ERROR=1

python navsim/planning/script/run_mean_std.py\
    agent=diffusiondrive_agent\
    experiment_name=run_diffusiondrive_agent_mean_std\
    train_test_split=navtrain\
    split=trainval\
    use_cache_without_dataset=True \
    cache_path="${NAVSIM_EXP_ROOT}/training_cache/" \
    force_cache_computation=False 