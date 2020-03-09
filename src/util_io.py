from os.path import expanduser, join

def pform(path, *names, sep= ''):
    """formats a path as `path` followed by `names` joined with `sep`."""
    return join(expanduser(path), sep.join(map(str, names)))
