"""
General networks for pytorch.

Algorithm-specific networks should go else-where.
"""
from rlkit.torch.networks.basic import (
    Clamp,
    ConcatTuple,
    Detach,
    Flatten,
    FlattenEach,
    Reshape,
    Split,
)
from rlkit.torch.networks.cnn import CNN, BasicCNN, CNNPolicy, MergedCNN
from rlkit.torch.networks.dcnn import DCNN, TwoHeadDCNN
from rlkit.torch.networks.feat_point_mlp import FeatPointMlp
from rlkit.torch.networks.image_state import ImageStatePolicy, ImageStateQ
from rlkit.torch.networks.linear_transform import LinearTransform
from rlkit.torch.networks.mlp import (
    ConcatMlp,
    ConcatMultiHeadedMlp,
    Mlp,
    MlpPolicy,
    MlpQf,
    MlpQfWithObsProcessor,
    TanhMlpPolicy,
)
from rlkit.torch.networks.normalization import LayerNorm
from rlkit.torch.networks.pretrained_cnn import PretrainedCNN
from rlkit.torch.networks.two_headed_mlp import TwoHeadMlp

__all__ = [
    "Clamp",
    "ConcatMlp",
    "ConcatMultiHeadedMlp",
    "ConcatTuple",
    "BasicCNN",
    "CNN",
    "CNNPolicy",
    "DCNN",
    "Detach",
    "FeatPointMlp",
    "Flatten",
    "FlattenEach",
    "LayerNorm",
    "LinearTransform",
    "ImageStatePolicy",
    "ImageStateQ",
    "MergedCNN",
    "Mlp",
    "PretrainedCNN",
    "Reshape",
    "Split",
    "TwoHeadDCNN",
    "TwoHeadMlp",
]
