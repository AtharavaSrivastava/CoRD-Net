"""
models/
=======
CoRD-Net module library.

Import the top-level model::

    from models.drpnet import DRPNet
    model = DRPNet(cfg.model)

Or import individual components for unit testing::

    from models.roi import DRPBlock
    from models.pgr import PGRModule
"""

from models.drpnet import DRPNet
from models.fgbf import FineGrainedBoundaryFeatureModule

__all__ = ["DRPNet", "FineGrainedBoundaryFeatureModule"]
