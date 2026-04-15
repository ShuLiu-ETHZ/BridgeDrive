from typing import Tuple
from pathlib import Path
import logging

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.dataset import CacheOnlyDataset, Dataset
from navsim.planning.training.agent_lightning_module import AgentLightningModule

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for training an agent.
    :param cfg: omegaconf dictionary
    """

    pl.seed_everything(cfg.seed, workers=True)
    logger.info(f"Global Seed set to {cfg.seed}")

    logger.info(f"Path where all results are stored: {cfg.output_dir}")

    logger.info("Building Agent")
    agent: AbstractAgent = instantiate(cfg.agent)

    logger.info("Using cached data without building SceneLoader")
    assert (
        not cfg.force_cache_computation
    ), "force_cache_computation must be False when using cached data without building SceneLoader"
    assert (
        cfg.cache_path is not None
    ), "cache_path must be provided when using cached data without building SceneLoader"
    train_data = CacheOnlyDataset(
        cache_path=cfg.cache_path,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        log_names=cfg.train_logs,
    )


    all_traj = []
    idx = 0
    for _, targets in tqdm(train_data):
        all_traj.append(
            targets["trajectory"][...,:2].numpy()
        )
        idx += 1

        

    
    mean = np.mean(all_traj, 0)
    std = np.std(all_traj, 0)

    np.save("traj_mean.npy", mean)
    np.save("traj_std.npy", std)


    print(mean)
    print(std)


if __name__ == "__main__":
    main()
