from YamlObj import YamlObj
from dataclasses import dataclass


@dataclass
class System(YamlObj):
    execution_mode: str           # "ml" | "hpc"
    fram_schd: str   # framework (ML) or scheduler (HPC)
    run_id: str

    VALID_FRAMEWORKS = {"pytorch", "tensorflow"}
    VALID_SCHEDULERS = {"slurm", "pbs", "lsf"}

    def validate(self) -> bool:
        # Check execution_mode
        if self.execution_mode not in {"ml", "hpc"}:
            raise ValueError("execution_mode must be 'ml' or 'hpc'")

        if not self.fram_schd:
            raise ValueError("fram_schd is required for both ML and HPC modes")

        # ML mode validations
        if self.execution_mode == "ml":
            if self.fram_schd not in self.VALID_FRAMEWORKS:
                raise ValueError(
                    f"Invalid framework '{self.fram_schd}' for ML mode. "
                    f"Allowed: {self.VALID_FRAMEWORKS}"
                )

        # HPC mode validations
        if self.execution_mode == "hpc":
            if self.fram_schd not in self.VALID_SCHEDULERS:
                raise ValueError(
                    f"Invalid scheduler '{self.fram_schd}' for HPC mode. "
                    f"Allowed: {self.VALID_SCHEDULERS}"
                )

        return True
