from .basemodel import BaseModel
from .gcn import GCN
from .savn import SAVN
from .protomodel import ProtoModel

__all__ = ["BaseModel", "GCN", "SAVN", "ProtoModel"]

variables = locals()
