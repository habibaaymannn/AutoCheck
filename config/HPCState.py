
class HPCState:
    name: str
    type: str
    source: str

    VALID_TYPES = {"int", "float", "str", "bool"}

    def __init__(self, name, type_, source):
        self.name = name
        self.type = type_
        self.source = source
        self.validate()

    def validate(self) -> bool:

        # Name validation
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("HPCState name must be a non-empty string")

        # Type validation
        if not isinstance(self.type, str):
            raise ValueError("HPCState type must be a string")
        self.type = self.type.lower()
        if self.type not in self.VALID_TYPES:
            raise ValueError(
                f"Invalid type '{self.type}'. "
                f"Allowed types: {self.VALID_TYPES}"
            )

        # Source validation
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("HPCState source must be a non-empty string")

        return True
