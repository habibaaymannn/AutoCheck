from enum import Enum


class ExecutionMode(str, Enum):
    ML = "ml"
    HPC = "hpc"


class MLFramework(str, Enum):
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    SKLEARN = "sklearn"


class HPCScheduler(str, Enum):
    SLURM = "slurm"
    PBS = "pbs"
    LSF = "lsf"


class CheckpointMethod(str, Enum):
    TIME = "time"
    ITERATION = "iteration"
    EPOCH = "epoch"
    BATCH = "batch"
    STEP = "step"


class StateType(str, Enum):
    INT = "int"
    FLOAT = "float"
    STR = "str"
    BOOL = "bool"
