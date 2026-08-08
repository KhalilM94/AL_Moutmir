"""Building the heavy data payload once, outside the study.

LightningConfigFactory rebuilds a sequence bundle from the raw CSVs whenever `data` does not already
carry one. Doing that per trial would dwarf the training it is meant to feed, so the payload is
built once here and handed to every trial.
"""

from __future__ import annotations

from typing import Any, Mapping

from yg_eo_soilnet.models.config_fatories.lightning_config_factory import LightningConfigFactory

PAYLOAD_KEYS = {"sequence": "sequence_bundle", "graph": "spatiotemporal_graph"}


def build_lightning_input(
    entry: str,
    spec: Mapping[str, Any],
    config: Any,
    split_data: Mapping[str, Any],
    *,
    logger: Any = None,
    data_manager: Any = None,
) -> dict[str, Any]:
    """`split_data` plus the datamodule payload the entry's `input_kind` needs."""
    data = dict(split_data)
    input_kind = LightningConfigFactory._input_kind(spec)
    payload_key = PAYLOAD_KEYS.get(input_kind)
    if payload_key is None:
        raise ValueError(
            f"Registry entry {entry!r} declares input_kind {input_kind!r}; expected one of "
            f"{', '.join(sorted(PAYLOAD_KEYS))}."
        )
    if data.get(payload_key) is not None:
        return data

    factory = LightningConfigFactory({entry: dict(spec)}, config, logger=logger, data_manager=data_manager)
    builder = factory._build_sequence_bundle if input_kind == "sequence" else factory._build_spatiotemporal_graph
    data[payload_key] = builder(spec)
    return data
