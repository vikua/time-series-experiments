from enum import Enum

import attr

from ._base import Stack
from ._stacks import (
    DRESSStack,
    ParallelStack,
    NoResidualStack,
    LastForwardStack,
    NoResidualLastForwardStack,
    ResidualInputStack,
)


@attr.s
class StackDef(object):
    stack_type = attr.ib()
    block_types = attr.ib()
    block_units = attr.ib()
    block_theta_units = attr.ib()
    block_layers = attr.ib(default=4)
    block_activation = attr.ib(default="relu")
    block_kernel_initializer = attr.ib(default="glorot_uniform")
    block_bias_initializer = attr.ib(default="zeros")
    block_kernel_regularizer = attr.ib(default=None)
    block_bias_regularizer = attr.ib(default=None)
    block_kernel_constraint = attr.ib(default=None)
    block_bias_constraint = attr.ib(default=None)

    def get_params(self):
        return attr.asdict(self, filter=lambda attr, value: attr.name != "stack_type")


class StackTypes(Enum):
    NBEATS_DRESS = 0
    PARALLEL = 1
    NO_RESIDUAL = 2
    LAST_FORWARD = 3
    NO_RESIDUAL_LAST_FORWARD = 4
    RESIDUAL_INPUT = 5


STACKS = {
    StackTypes.NBEATS_DRESS: DRESSStack,
    StackTypes.PARALLEL: ParallelStack,
    StackTypes.NO_RESIDUAL: NoResidualStack,
    StackTypes.LAST_FORWARD: LastForwardStack,
    StackTypes.NO_RESIDUAL_LAST_FORWARD: NoResidualLastForwardStack,
    StackTypes.RESIDUAL_INPUT: ResidualInputStack,
}


__all__ = [
    "StackDef",
    "StackTypes",
    "STACKS",
    "Stack",
    "DRESSStack",
    "DRESSStack",
    "ParallelStack",
    "NoResidualStack",
    "LastForwardStack",
    "NoResidualLastForwardStack",
    "ResidualInputStack",
]
