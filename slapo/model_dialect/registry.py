# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Framework model dialect registration."""

DIALECTS = {"pipeline": {}, "log_parser": {}}


def register_model_dialect(target, cls_type):
    """Register a framework model dialect."""
    if cls_type not in DIALECTS:
        raise ValueError(f"Only support {DIALECTS.keys()}, but got {cls_type}")

    def decorator(dialect_cls):
        if "target" in DIALECTS[cls_type]:
            raise ValueError(
                f"Target {target} already registered for {cls_type} model dialects"
            )
        DIALECTS[cls_type][target] = dialect_cls
        return dialect_cls

    return decorator


def get_all_dialects(cls_type):
    """Get all registered framework model dialects."""
    if cls_type not in DIALECTS:
        raise ValueError(f"Only support {DIALECTS.keys()}, but got {cls_type}")
    return DIALECTS[cls_type]


def get_dialect_cls(cls_type, target):
    """Get the framework model dialect class."""
    if cls_type not in DIALECTS:
        raise ValueError(f"Only support {DIALECTS.keys()}, but got {cls_type}")
    if target not in DIALECTS[cls_type]:
        raise ValueError(
            f"Target {target} not registered for {cls_type} model dialects"
        )
    return DIALECTS[cls_type][target]
