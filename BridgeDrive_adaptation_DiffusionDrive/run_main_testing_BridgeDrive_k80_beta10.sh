export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/datashare/DiffusionDrive/download/maps"
export NAVSIM_EXP_ROOT="/datashare/DiffusionDrive/exp"
export NAVSIM_DEVKIT_ROOT="/data/workspace/shuliu/BridgeDrive_adaptation_DiffusionDrive/navsim"
export OPENSCENE_DATA_ROOT="/datashare/DiffusionDrive/download"

export CUDA_VISIBLE_DEVICES=3
export HYDRA_FULL_ERROR=1

# CKPT=/datashare/DiffusionDrive/exp/training_diffusiondrive_agent/2025.11.21.00.27.27_k80_beta10/lightning_logs/version_0/checkpoints/epoch99s33300.ckpt
CKPT=/change/to/your/ckpt/path

# update to your paths
python /data/workspace/shuliu/BridgeDrive_adaptation_DiffusionDrive/navsim/planning/script/run_pdm_score.py \
    train_test_split=navtest \
    agent=bridgedrive_agent_k80_beta10 \
    worker=ray_distributed \
    agent.checkpoint_path=$CKPT \
    experiment_name=bridgedrive_agent_eval