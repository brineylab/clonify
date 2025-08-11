from importlib import import_module

# Import version
from .version import __version__

# Import the implementation submodule *once* so that it is available for the CLI and
# any other internal imports.  Importing the submodule normally would cause Python
# to set an attribute named `clonify` **on this package** that points to the
# *module* object, unintentionally shadowing the public `clonify` callable defined
# below.  To avoid this we immediately overwrite that attribute with the function
# we actually want to expose.
_clonify_mod = import_module(".clonify", __name__)


def clonify(*args, **kwargs):
    """Public clonify API.

    This is a very thin wrapper around :pyfunc:`clonify.clonify.clonify` that
    simply forwards all positional and keyword arguments.  Defining it here
    (instead of re-exporting the submodule attribute directly) ensures that the
    symbol `clonify` on the *package* always refers to a **callable**, even after
    the `clonify` submodule has been imported elsewhere in the process.
    """

    return _clonify_mod.clonify(*args, **kwargs)


# Overwrite the module attribute that was automatically created during the
# submodule import so that `import clonify; clonify.clonify(...)` continues to
# work and still resolves to the callable, not to the submodule object.
globals()["clonify"] = clonify

__all__ = ["clonify", "__version__"]
