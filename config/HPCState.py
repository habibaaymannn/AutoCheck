
class HPCState:
    _name: str
    _type: str
    _source: str

    def __init__(self, name, __type, source):
        self._name = name
        self._type = __type
        self._source = source

    def validate(self)->bool:
        pass
