export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/data/datasets/driving/DiffusionDrive/download/maps"
export NAVSIM_EXP_ROOT="/data/datasets/driving/DiffusionDrive/exp"
export NAVSIM_DEVKIT_ROOT="/data/workspace/shuliu/DiffusionDrive/navsim"
export OPENSCENE_DATA_ROOT="/data/datasets/driving/DiffusionDrive/download"

python navsim/planning/script/run_dataset_caching.py\
 agent=diffusiondrive_agent\
 experiment_name=training_diffusiondrive_agent\
 train_test_split=navtrain\
 cache_path="${NAVSIM_EXP_ROOT}/training_cache_cmd_speed/"