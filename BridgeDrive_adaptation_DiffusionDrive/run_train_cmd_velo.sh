export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/data/datasets/driving/DiffusionDrive/download/maps"
export NAVSIM_EXP_ROOT="/data/datasets/driving/DiffusionDrive/exp"
export NAVSIM_DEVKIT_ROOT="/data/workspace/shuliu/DiffusionDrive/navsim"
export OPENSCENE_DATA_ROOT="/data/datasets/driving/DiffusionDrive/download"

export HYDRA_FULL_ERROR=1

export CUDA_VISIBLE_DEVICES=4,5

python /data/workspace/shuliu/DiffusionDrive/navsim/planning/script/run_training.py \
        agent=diffusiondrive_agent \
        experiment_name=training_diffusiondrive_agent  \
        train_test_split=navtrain  \
        split=trainval   \
        trainer.params.max_epochs=100 \
        dataloader.params.batch_size=256 \
        cache_path="${NAVSIM_EXP_ROOT}/training_cache_cmd_vel/" \
        use_cache_without_dataset=True  \
        force_cache_computation=False 