import os


def _version():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    import git
    g = git.cmd.Git(dir_path)
    g.fetch()
    return g.describe('--tags')


if __name__ == "__main__":
    print(_version())
else:
    __version__ = _version()
