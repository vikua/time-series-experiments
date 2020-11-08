from enum import Enum

from ._base import Block
from ._generic import GenericBlock
from ._trend import TrendBlock
from ._seasonal import SeasonalBlock


class BlockTypes(Enum):
    GENERIC = 1
    TREND = 2
    SEASONAL = 3


BLOCKS = {
    BlockTypes.GENERIC: GenericBlock,
    BlockTypes.TREND: TrendBlock,
    BlockTypes.SEASONAL: SeasonalBlock,
}


__all__ = [
    "BlockTypes",
    "BLOCKS",
    "Block",
    "GenericBlock",
    "TrendBlock",
    "SeasonalBlock",
]
