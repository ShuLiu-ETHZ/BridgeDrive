export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/datashare/DiffusionDrive/download/maps"
export NAVSIM_EXP_ROOT="/datashare/DiffusionDrive/exp"
export NAVSIM_DEVKIT_ROOT="/data/workspace/shuliu/BridgeDrive_adaptation_DiffusionDrive/navsim"
export OPENSCENE_DATA_ROOT="/datashare/DiffusionDrive/download"

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=2 # One-GPU training leads to better performance

# update to your paths
python /data/workspace/shuliu/BridgeDrive_adaptation_DiffusionDrive/navsim/planning/script/run_training.py \ 
        agent=bridgedrive_agent_k80_beta10 \
        experiment_name=training_bridgedrive_agent  \
        train_test_split=navtrain  \
        split=trainval   \
        trainer.params.max_epochs=100 \
        dataloader.params.batch_size=256 \
        cache_path="${NAVSIM_EXP_ROOT}/training_cache/" \
        use_cache_without_dataset=True  \
        force_cache_computation=False 