from .basemodel import BaseModel
from .gcn import GCN
from .savn import SAVN
from .relnet_model import RelnetModel

__all__ = ["BaseModel", "GCN", "SAVN", "RelnetModel"]

variables = locals()
