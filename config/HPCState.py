class HPCState:
    _name: str
    _type: str
    _source: str

    VALID_TYPES = {"int", "float", "str", "bool"}

    def __init__(self, name, __type, source):
        self._name = name
        self._type = __type
        self._source = source
        self.validate()

    def validate(self) -> bool:

        # Name validation
        if not isinstance(self._name, str) or not self._name.strip():
            raise ValueError("HPCState name must be a non-empty string")

        # Type validation
        if not isinstance(self._type, str):
            raise ValueError("HPCState type must be a string")
        self._type = self._type.lower()
        if self._type not in self.VALID_TYPES:
            raise ValueError(
                f"Invalid type '{self._type}'. "
                f"Allowed types: {self.VALID_TYPES}"
            )

        # Source validation
        if not isinstance(self._source, str) or not self._source.strip():
            raise ValueError("HPCState source must be a non-empty string")

        return True
