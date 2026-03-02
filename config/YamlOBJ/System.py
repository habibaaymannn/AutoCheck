from config.YamlOBJ.YamlObj import YamlObj
from dataclasses import dataclass
from enums import ExecutionMode, MLFramework, HPCScheduler

@dataclass
class System(YamlObj):
    execution_mode: str           # "ml" | "hpc"
    fram_schd: str   # framework (ML) or scheduler (HPC)
    run_id: str

    def validate(self) -> bool:
        # Check execution_mode
        if not isinstance(self.execution_mode,str):
            raise ValueError("execution_mode must be a string")
        self.execution_mode=self.execution_mode.lower()

        allowed_modes={e.value for e in ExecutionMode}
        if self.execution_mode not in allowed_modes:
            raise ValueError(f"execution_mode must be one of {allowed_modes}")

        if not self.fram_schd:
            raise ValueError("fram_schd is required for both ML and HPC modes")
        self.fram_schd = self.fram_schd.lower()

        # ML mode validations
        if self.execution_mode == ExecutionMode.ML.value:
            allowed_frameworks={e.value for e in MLFramework}
            if self.fram_schd not in allowed_frameworks:
                raise ValueError(
                    f"Invalid framework '{self.fram_schd}' for ML mode. "
                    f"Allowed: {allowed_frameworks}"
                )

        # HPC mode validations
        if self.execution_mode == ExecutionMode.HPC.value:
            allowed_schd={e.value for e in HPCScheduler}
            if self.fram_schd not in allowed_schd:
                raise ValueError(
                    f"Invalid scheduler '{self.fram_schd}' for HPC mode. "
                    f"Allowed: {allowed_schd}"
                )

        return True
