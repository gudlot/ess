import os


def _version():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        import git
        g = git.cmd.Git(dir_path)
        g.fetch()
        return g.describe('--tags')
    except (ImportError, git.exc.GitError):
        from . import _fixed_version
        return _fixed_version.__version__


if __name__ == "__main__":
    print(_version())
else:
    __version__ = _version()
