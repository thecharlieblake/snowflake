from enum import Enum, unique


@unique
class NetworkTypes(Enum):
    fixedsizenet = 1
    nervenet = 2
    nervenetpp = 3
