from functools import singledispatch
import argparse

@singledispatch
def Namespace(ob):
    return ob

@Namespace.register(dict)
def _wrap_dict(ob):
    return argparse.Namespace(**{k: Namespace(v) for k, v in ob.items()})

@Namespace.register(list)
def _wrap_list(ob):
    return [Namespace(v) for v in ob]