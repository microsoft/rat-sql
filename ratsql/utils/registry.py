import collections
import collections.abc
import inspect
import sys


_REGISTRY = collections.defaultdict(dict)


def register(kind, name):
    kind_registry = _REGISTRY[kind]

    def decorator(obj):
        if name in kind_registry:
            raise LookupError(f'{name} already registered as kind {kind}')
        kind_registry[name] = obj
        return obj

    return decorator


def lookup(kind, name):
    if isinstance(name, collections.abc.Mapping):
        name = name['name']

    if kind not in _REGISTRY:
        raise KeyError(f'Nothing registered under "{kind}"')
    return _REGISTRY[kind][name]


def construct(kind, config, unused_keys=(), **kwargs):
    return instantiate(
            lookup(kind, config),
            config,
            unused_keys + ('name',),
            **kwargs)


def instantiate(invocable, config, unused_keys=(), **kwargs):
    merged = {**config, **kwargs}
    signature = inspect.signature(invocable)

    if hasattr(invocable, '__init__'):
        # to avoid inspecting ctor of parent class (if exists) instead of the target class's ctor
        params = dict(inspect.signature(invocable.__init__).parameters)
        params.pop('self', None)
    else:
        params = dict(inspect.signature(invocable).parameters)

    for name, param in params.items():
        if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.VAR_POSITIONAL):
            raise ValueError(f'Unsupported kind for param {name}: {param.kind}')

    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return invocable(**merged)

    missing = {}
    for key in list(merged.keys()):
        if key not in params:
            if key not in unused_keys:
                missing[key] = merged[key]
            merged.pop(key)
    if missing:
        print(f'WARNING {invocable}: superfluous {missing}', file=sys.stderr)
    return invocable(**merged)
