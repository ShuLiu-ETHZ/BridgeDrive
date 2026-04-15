export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/datashare/DiffusionDrive/download/maps"
export NAVSIM_EXP_ROOT="/datashare/DiffusionDrive/exp"
export NAVSIM_DEVKIT_ROOT="/data/workspace/shuliu/DiffusionDrive/navsim"
export OPENSCENE_DATA_ROOT="/datashare/DiffusionDrive/download"

export CUDA_VISIBLE_DEVICES=3
export HYDRA_FULL_ERROR=1

CKPT=/datashare/DiffusionDrive/exp/training_diffusiondrive_agent/2025.07.08.12.24.19/lightning_logs/version_0/last.ckpt


python /data/workspace/shuliu/DiffusionDrive/navsim/planning/script/run_instances.py \
    train_test_split=navtest \
    agent=diffusiondrive_agent \
    worker=ray_distributed \
    agent.checkpoint_path=$CKPT \
    experiment_name=diffusiondrive_agent_eval


