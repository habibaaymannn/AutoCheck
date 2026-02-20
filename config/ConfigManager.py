import yaml
from typing import List, Dict, Callable, Any

from config.YamlOBJ.YamlObj import YamlObj
from config.YamlOBJ.Checkpoint import Checkpoint
from config.YamlOBJ.System import System
from config.YamlOBJ.Notify import Notify
from config.YamlOBJ.HPC import HPC
from config.YamlOBJ.HPCState import HPCState
from config.YamlOBJ.ML import ML
from config.YamlOBJ.enum import ExecutionMode


class ConfigParseError(Exception):
    pass


class ConfigValidationError(Exception):
    pass

# Required for every mode
COMMON_REQUIRED = {"system","checkpoint"}

# Required sections per mode
MODE_REQUIRED = {
    "ml":  {"ml_model"},
    "hpc": {"hpc"},
}

# Forbidden sections per mode
MODE_FORBIDDEN = {
    ExecutionMode.ML.value: {"hpc"},
    ExecutionMode.HPC.value: {"ml_model"},
}

# Optional across all modes
OPTIONAL_SECTIONS = {"notify"}

# Full known sections
KNOWN_SECTIONS = COMMON_REQUIRED | OPTIONAL_SECTIONS | {"ml_model", "hpc"}


class ConfigManager:

    def __init__(self):
        self.configs: List[YamlObj] = []

    def parse(self, yaml_path: str) -> None:
        with open(yaml_path, "r",encoding="utf-8") as f:
            raw: dict = yaml.safe_load(f)

        if raw is None:
            raise ConfigParseError("YAML file is empty")
        if not isinstance(raw, dict):
            raise ConfigParseError(f"Expected a YAML mapping at the top level, got {type(raw)}")

        self.configs.clear()

        # step 1: unkown section detection
        unknown_sections=set(raw.keys())-KNOWN_SECTIONS
        if unknown_sections:
            raise ConfigParseError(
                f"Unknown top-level section(s): {sorted(unknown_sections)}. "
                f"Allowed sections: {sorted(KNOWN_SECTIONS)}"
            )

        # Step 2: system is always required and parsed first
        if "system" not in raw:
            raise ConfigParseError("Missing required 'system' section in config")
        
        #step 3: Type Checks before parsing
        for section in raw:
            self._ensure_mapping(raw, section)

        
        #Parse system first to determine mode
        system = System(**raw["system"])
        self.configs.append(system)

        mode= system.execution_mode.value if hasattr(system.execution_mode,"value") else system.execution_mode

        if mode not in MODE_REQUIRED:
            raise ConfigParseError(f"Unsupported execution_mode '{mode}'")

        # Step 4: check all required sections for this mode are present
        required_sections = COMMON_REQUIRED | MODE_REQUIRED[mode]
        missing = required_sections - raw.keys()
        if missing:
            raise ConfigParseError(
                f"Missing required section(s) for '{mode}' mode: {sorted(missing)}"
            )
        forbidden=MODE_FORBIDDEN[mode] & set(raw.keys())
        if forbidden:
            raise ConfigParseError(
                f"Forbidden section(s) for '{mode}' mode: {sorted(forbidden)}"
            )
        
        # Parse sections in deterministic order
        section_parsers: Dict[str, Callable[[Dict[str, Any], System], YamlObj]] = {
            "ml_model": self.parse_ml,
            "checkpoint": self.parse_checkpoint,
            "notify": self.parse_notify,
            "hpc": self.parse_hpc,
        }

        allowed_sections = required_sections | OPTIONAL_SECTIONS
        for key, parser in section_parsers.items():
            if key not in raw:
                continue
            if key not in allowed_sections:
                continue
            self.configs.append(section_parsers[key](raw[key], system))

    def parse_ml(self, data: dict, system: System) -> ML:
        if "name" not in data:
            raise ConfigParseError("Missing required field 'ml_model.name'")
        return ML(name=data["name"], ml_system=system)

    def parse_checkpoint(self, data: dict, system: System) -> Checkpoint:
        return Checkpoint(**data)

    def parse_notify(self, data: dict, system: System) -> Notify:
        return Notify(**data)

    def parse_hpc(self, data: dict, system: System) -> HPC:
        tracked_raw = data.get("tracked_states", [])
        if not isinstance(tracked_raw, list):
            raise ConfigParseError("'hpc.tracked_states' must be a list")

        tracked_states = []
        for i, s in enumerate(tracked_raw):
            if not isinstance(s, dict):
                raise ConfigParseError(f"'hpc.tracked_states[{i}]' must be a mapping/object")
            for field in ("name", "type", "source"):
                if field not in s:
                    raise ConfigParseError(f"Missing required field 'hpc.tracked_states[{i}].{field}'")
            tracked_states.append(HPCState(name=s["name"], type_=s["type"], source=s["source"]))

        return HPC(tracked_states=tracked_states)

    def validate(self) -> bool:
        if not self.configs:
            raise ConfigValidationError("No configs loaded - call parse() first")

        errors:List[str]=[]

        #step 5: Aggregate per-object validation
        for obj in self.configs:
            try:
                obj.validate()
            except ValueError as e:
                errors.append(f"{obj.__class__.__name__}: {e}")

        #step 6:Cross-object validation rules
        try:
            checkpoint=self.get(Checkpoint)
            if checkpoint.interval >= checkpoint.max_session_time:
                errors.append(
                    "Checkpoint: interval must be smaller than max_session_time"
                )
            if checkpoint.safety_buffer_seconds >= checkpoint.max_session_time:
                errors.append(
                    "Checkpoint: safety_buffer_seconds must be smaller than max_session_time"
                )   
        except KeyError:
            errors.append("Checkpoint: missing checkpoint config")

        if errors:
            raise ConfigValidationError("Configuration validation failed:\n- " + "\n- ".join(errors))
        return True


    def get(self, cls: type) -> YamlObj:
        """Return the first config object that is an instance of *cls*."""
        for obj in self.configs:
            if isinstance(obj, cls):
                return obj
        raise KeyError(f"No config of type '{cls.__name__}' found")

    @property
    def mode(self) -> str:
        mode= self.get(System).execution_mode
        return mode.value if hasattr(mode,"value")else mode
    
    @staticmethod
    def _ensure_mapping(raw: dict, section: str) -> None:
        if not isinstance(raw.get(section), dict):
            raise ConfigParseError(f"Section '{section}' must be a mapping/object")