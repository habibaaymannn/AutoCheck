import yaml
from typing import List
from config.YamlOBJ.YamlObj import YamlObj
from config.YamlOBJ.Checkpoint import Checkpoint
from config.YamlOBJ.System import System
from config.YamlOBJ.Notify import Notify
from config.YamlOBJ.HPC import HPC
from config.YamlOBJ.HPCState import HPCState
from config.YamlOBJ.ML import ML


# Required for every mode
COMMON_REQUIRED = {"checkpoint"}

# Required sections per mode
MODE_REQUIRED = {
    "ml":  {"ml_model"},
    "hpc": {"hpc"},
}

# Optional across all modes
OPTIONAL_SECTIONS = {"notify"}


class ConfigManager:

    def __init__(self):
        self.configs: List[YamlObj] = []

    def parse(self, yaml_path: str) -> None:
        with open(yaml_path, "r") as f:
            raw: dict = yaml.safe_load(f)

        if not isinstance(raw, dict):
            raise ValueError(f"Expected a YAML mapping at the top level, got {type(raw)}")

        self.configs.clear()

        # Step 1: system is always required and parsed first
        if "system" not in raw:
            raise ValueError("Missing required 'system' section in config")

        system = System(**raw["system"])
        self.configs.append(system)
        mode = system.execution_mode.lower()  # "ml" or "hpc"

        # Step 2: check all required sections for this mode are present
        required_sections = COMMON_REQUIRED | MODE_REQUIRED[mode]
        missing = required_sections - raw.keys()
        if missing:
            raise ValueError(f"Missing required section(s) for '{mode}' mode: {missing}")

        # Step 3: parse sections — required ones are guaranteed present, optional are skipped if absent
        allowed_sections = required_sections | OPTIONAL_SECTIONS
        section_parsers = {
            "ml_model":   self.parse_ml,
            "checkpoint": self.parse_checkpoint,
            "notify":     self.parse_notify,
            "hpc":        self.parse_hpc,
        }

        for key, parser in section_parsers.items():
            if key not in raw:
                continue  # only possible for optional sections at this point

            if key not in allowed_sections:
                print(f"[ConfigManager] Ignoring '{key}' section (not used in '{mode}' mode)")
                continue

            self.configs.append(parser(raw[key], system))

    def parse_ml(self, data: dict, system: System) -> ML:
        return ML(name=data["name"], ml_system=system)

    def parse_checkpoint(self, data: dict, system: System) -> Checkpoint:
        return Checkpoint(**data)

    def parse_notify(self, data: dict, system: System) -> Notify:
        return Notify(**data)

    def parse_hpc(self, data: dict, system: System) -> HPC:
        tracked_states = [
            HPCState(name=s["name"], type_=s["type"], source=s["source"])
            for s in data.get("tracked_states", [])
        ]
        return HPC(tracked_states=tracked_states)

    def validate(self) -> bool:
        if not self.configs:
            raise ValueError("No configs loaded — call parse() first")

        for obj in self.configs:
            obj.validate()

        return True

    def get(self, cls: type) -> YamlObj:
        """Return the first config object that is an instance of *cls*."""
        for obj in self.configs:
            if isinstance(obj, cls):
                return obj
        raise KeyError(f"No config of type '{cls.__name__}' found")

    @property
    def mode(self) -> str:
        return self.get(System).execution_mode