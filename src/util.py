from collections.abc import Mapping

class Record(Mapping):
    """a `dict`-like type whose instances are partial finite mappings from
    attribute keys to arbitrary values.
    like a dict, a record is transparent about its content.
    unlike a dict, a record can access its content as object
    attributes without the hassle of string quoting.
    """

    def __init__(self, *records, **entries):
        for rec in records:
            for key, val in rec.items():
                setattr(self, key, val)
        for key, val in entries.items():
            setattr(self, key, val)

    def __repr__(self):
        return repr(self.__dict__)

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)
