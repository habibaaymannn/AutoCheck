from dataclasses import dataclass
from YamlObj import YamlObj


@dataclass
class System(YamlObj):
    VALID_FRAMEWORKS = {"pytorch", "tensorflow"}
    VALID_SCHEDULERS = {"slurm", "pbs", "lsf"}
    execution_mode: str
    fram_schd: str
    run_id: str

    def validate(self) -> bool:
        # execution_mode must be either "ml" or "hpc"
        if self.execution_mode not in {"ml", "hpc"}:
            raise ValueError("execution_mode must be 'ml' or 'hpc'")

        # fram_schd is required for both modes
        if not self.fram_schd:
            raise ValueError("fram_schd is required for both ML and HPC modes")

        # ML mode: fram_schd must be a valid framework
        if self.execution_mode == "ml":
            if self.fram_schd not in self.VALID_FRAMEWORKS:
                raise ValueError(
                    f"Invalid framework '{self.fram_schd}' for ML mode. "
                    f"Allowed: {self.VALID_FRAMEWORKS}"
                )

        # HPC mode: fram_schd must be a valid scheduler
        if self.execution_mode == "hpc":
            if self.fram_schd not in self.VALID_SCHEDULERS:
                raise ValueError(
                    f"Invalid scheduler '{self.fram_schd}' for HPC mode. "
                    f"Allowed: {self.VALID_SCHEDULERS}"
                )

        # run_id must be a non-empty string
        if not self.run_id:
            raise ValueError("run_id is required")

        return True