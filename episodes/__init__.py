from .basic_episode import BasicEpisode
from .test_val_episode_ithor import IthorTestValEpisode
from .test_val_episode_robothor import RobothorTestValEpisode
__all__ = [
    'BasicEpisode',
    'IthorTestValEpisode',
    'RobothorTestValEpisode'
]

# All models should inherit from BasicEpisode
variables = locals()