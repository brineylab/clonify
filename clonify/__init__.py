def clonify(*args, **kwargs):
    # Thin wrapper to forward all args/kwargs to the implementation,
    # ensuring new keyword-only options are accepted without import-time binding issues.
    from .clonify import clonify as _impl

    return _impl(*args, **kwargs)


__all__ = ["clonify"]
