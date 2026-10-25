"""Models: energy layer ET / Forced and the task models around it (Phase C.4)."""
from models.modeling_base import EnergyLoopConfig, EnergyLoopModel, LatentEnergyLayer, build_model, list_models
import models.tasks  # noqa: F401  (registers inverse, lowrank, addition, sudoku_rrn)
