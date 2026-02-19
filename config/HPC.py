from dataclasses import dataclass, field
from typing import List
from AutoCheck.config.HPCState import HPCState
from AutoCheck.config.YamlObj import YamlObj


@dataclass
class HPC(YamlObj):
    tracked_states: List[HPCState] = field(default_factory=list)

    def validate(self) -> bool:
        # Must have at least one tracked state
        if not self.tracked_states:
            raise ValueError("HPC must define at least one tracked_state")

        # Ensure all elements are HPCState
        for state in self.tracked_states:
            if not isinstance(state, HPCState):
                raise ValueError(
                    "All tracked_states must be instances of HPCState"
                )

        # Prevent duplicate state names
        names = [state.name for state in self.tracked_states]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate HPCState names are not allowed")

        return True
