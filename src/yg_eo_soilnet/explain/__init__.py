"""SHAP explainability for both training families.

The two entry points the logger uses:

* :func:`build_shap_results` dispatches to the backend explainer and returns one
  :class:`~yg_eo_soilnet.explain.result.ShapResult` per target;
* :func:`log_shap_artifacts` turns those into the ``explain/`` artifacts.

**Nothing in this package imports ``shap`` at module scope, and nothing outside it imports this
package at module scope.** ``shap`` pulls in numba and is slow to import, so a run with
``EXPLAIN_ENABLED: false`` must not pay for it - and, because the test suite runs with
``filterwarnings = ["error"]``, must not risk a warning from a library it never asked for.
``ChildRunLogger._log_shap_artifacts`` checks the switch before importing this module;
``tests/test_explain_switch.py`` asserts ``"shap" not in sys.modules`` after a disabled run.
"""

from __future__ import annotations

from yg_eo_soilnet.artifacts import ArtifactLayout, log_figure, log_json, log_parquet
from yg_eo_soilnet.explain.result import (
    LOG1P_X10,
    ORIGINAL_UNITS,
    STANDARDIZED_LOG1P,
    ShapResult,
)

__all__ = [
    "ShapResult",
    "ORIGINAL_UNITS",
    "LOG1P_X10",
    "STANDARDIZED_LOG1P",
    "build_shap_results",
    "log_shap_artifacts",
]


def build_shap_results(*, config, backend: str, **payload) -> list[ShapResult]:
    """Explain one fitted model. ``backend`` is ``"sklearn"`` or ``"lightning"``."""
    if backend == "sklearn":
        from yg_eo_soilnet.explain.sklearn_explainer import sklearn_shap_results

        return sklearn_shap_results(config=config, **payload)

    if backend == "lightning":
        from yg_eo_soilnet.explain.lightning_explainer import lightning_shap_results

        return lightning_shap_results(config=config, **payload)

    raise ValueError(f"Unknown explain backend {backend!r}; expected 'sklearn' or 'lightning'")


def log_shap_artifacts(results: list[ShapResult], *, max_display: int = 25) -> dict:
    """Write the ``explain/`` artifacts for one child run and describe what was written.

    Per target: a beeswarm, a mean-|SHAP| bar, a per-block bar, and the COMPLETE per-sample value
    table. The plots are capped at ``max_display`` rows because the flat feature space runs to a
    hundred-plus entries once every band has its own row; the parquet is never capped, so the cap
    only ever limits the picture and not the data.

    Takes no target or model name: they used to be baked into every filename, which is exactly what
    made two runs share no artifact paths. The run's tags carry that identity instead.
    """
    from yg_eo_soilnet.explain.plots import shap_bar, shap_beeswarm, shap_block_bar

    written: dict = {"targets": [], "artifacts": []}

    # Filenames are stable so the same plot occupies the same path in every run and MLflow's compare
    # view can line them up. Only a run emitting SEVERAL targets nests them under explain/<target>/,
    # where the flat layout would have them overwrite each other.
    multi_target = len(results) > 1

    for result in results:
        artifact_path = ArtifactLayout.explain_path(result.target_name if multi_target else None)

        figures = [
            (shap_beeswarm(result, max_display=max_display), ArtifactLayout.SHAP_BEESWARM_FILE),
            (shap_bar(result, max_display=max_display), ArtifactLayout.SHAP_BAR_FILE),
            (shap_block_bar(result), ArtifactLayout.SHAP_BLOCK_BAR_FILE),
        ]
        for figure, filename in figures:
            log_figure(figure, filename, artifact_path)
            written["artifacts"].append(f"{artifact_path}/{filename}")

        log_parquet(result.to_frame(), ArtifactLayout.SHAP_VALUES_FILE, artifact_path)
        written["artifacts"].append(f"{artifact_path}/{ArtifactLayout.SHAP_VALUES_FILE}")

        log_json(result.summary(top_n=max_display), ArtifactLayout.SHAP_SUMMARY_FILE, artifact_path)
        written["artifacts"].append(f"{artifact_path}/{ArtifactLayout.SHAP_SUMMARY_FILE}")

        written["targets"].append(
            {
                "target": result.target_name,
                "output_space": result.output_space,
                "n_samples": result.n_samples,
                "n_features": result.n_features,
                "top_features": [
                    result.feature_names[index] for index in result.ranking()[:10]
                ],
                "blocks": result.block_mean_abs(),
            }
        )

    return written
