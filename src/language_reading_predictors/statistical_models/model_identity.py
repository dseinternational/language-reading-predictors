# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Stable identities of the computational graph, including its fitted inputs.

PyMC's human-readable model display abbreviates expressions as ``f(...)``. It
cannot distinguish addition from subtraction and is not a reuse fingerprint.
This module records the actual operators, their declared properties, ordered
inputs, constants and nested graphs. Random-generator state is excluded because
it determines a sample, not the model. The run contract binds the sampling seed.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any

import numpy as np
import pymc as pm
from pytensor.compile.sharedvalue import SharedVariable
from pytensor.graph.basic import Constant, Variable
from pytensor.graph.op import Op
from pytensor.graph.type import Type


def _class_name(value: object) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _value(value: Any) -> Any:
    """Encode graph properties without process-specific object representations."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return _value(value.item())
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            data = _value(value.tolist())
        else:
            data = hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
        return {"dtype": str(value.dtype), "shape": list(value.shape), "data": data}
    if isinstance(value, np.dtype):
        return str(value)
    if isinstance(value, slice):
        return {"slice": _value((value.start, value.stop, value.step))}
    if isinstance(value, Mapping):
        return sorted([(_json(_value(k)), _value(v)) for k, v in value.items()])
    if isinstance(value, (tuple, list)):
        return [_value(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_value(v) for v in value), key=_json)
    if isinstance(value, Enum):
        return {"class": _class_name(value), "value": _value(value.value)}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {"class": _class_name(value), "fields": _value(dataclasses.asdict(value))}
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    if isinstance(value, (Op, Type)) or hasattr(value, "__props__"):
        encoded = {
            "class": _class_name(value),
            "properties": {k: _value(getattr(value, k)) for k in (getattr(value, "__props__", ()) or ())},
        }
        if hasattr(value, "inner_outputs"):
            encoded["inner_graph"] = _graph(value.inner_outputs, value.inner_inputs)
        return encoded
    raise TypeError(f"Cannot fingerprint graph property {_class_name(value)}")


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _graph(outputs: Sequence[Variable], inputs: Sequence[Variable] = ()) -> dict[str, Any]:
    variables: dict[Variable, int] = {}
    nodes: list[dict[str, Any]] = []
    input_positions = {v: i for i, v in enumerate(inputs)}

    def visit(variable: Variable) -> int:
        if variable in variables:
            return variables[variable]
        record = {"type": _value(variable.type), "name": variable.name}
        if variable in input_positions:
            record["input"] = input_positions[variable]
        elif variable.owner is not None:
            owner = variable.owner
            record.update(
                inputs=[visit(v) for v in owner.inputs],
                op=_value(owner.op),
                output=owner.outputs.index(variable),
            )
        elif isinstance(variable, Constant):
            record["constant"] = _value(variable.data)
        elif isinstance(variable, SharedVariable):
            value = variable.get_value(borrow=True)
            if isinstance(value, (np.random.Generator, np.random.RandomState)):
                record["shared"] = {"random_generator": _class_name(value)}
            else:
                # Shape and dtype only. What is *inside* a shared node is the
                # design, not the structure, and hashing it here as well would
                # move ``structure_sha256`` whenever the data move — collapsing
                # the split ``design_sha256`` exists to provide, so a refusal
                # could no longer say whether the model code or only the data
                # changed (2026-09-05 review).
                array = np.asarray(value)
                record["shared"] = {"dtype": str(array.dtype), "shape": list(array.shape)}
        else:
            record["input"] = None
        index = len(nodes)
        variables[variable] = index
        nodes.append(record)
        return index

    roots = [visit(v) for v in outputs]
    return {"nodes": nodes, "outputs": roots}


def model_design_identity(model: pm.Model | None) -> dict[str, Any]:
    """Hash the built model and log density, or record why reuse must be refused."""
    try:
        if model is None:
            raise ValueError("no built model")
        names = sorted(model.named_vars)
        graph = _graph([model.named_vars[n] for n in names] + list(model.logp(sum=False)))
        structure = {"graph": graph, "coords": _value(model.coords), "dims": _value(model.named_vars_to_dims)}
        design = {
            n: _value(np.asarray(model.named_vars[n].get_value(borrow=True)))
            for n in names
            if isinstance(model.named_vars[n], SharedVariable)
        }
        return {
            "schema_version": 2,
            "structure_sha256": hashlib.sha256(_json(structure).encode()).hexdigest(),
            "design_sha256": hashlib.sha256(_json(design).encode()).hexdigest(),
            "design_arrays": list(design),
        }
    except Exception as exc:  # noqa: BLE001 - fitting is possible; reuse fails closed
        # Carries the schema version too, so a failure record is distinguishable
        # from a success one by shape as well as by its null hashes; the reuse
        # checks refuse on the missing hashes rather than comparing two failures.
        return {
            "schema_version": 2,
            "structure_sha256": None,
            "design_sha256": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
