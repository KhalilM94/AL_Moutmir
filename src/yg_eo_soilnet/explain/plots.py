"""SHAP figures, following the project's plotting convention.

As in plot_utils and hpo.plots: build a Figure, tight_layout it, and RETURN it - callers save and
close (yg_eo_soilnet.artifacts.log_figure does both). When there is nothing to draw, hand back a
figure carrying a message rather than None, so a caller never has to branch on the return value.

``import shap`` stays inside the functions so the module is cheap to import with SHAP disabled.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt
import numpy as np

from yg_eo_soilnet.explain.result import ShapResult


def _message_figure(message: str):
    figure, axes = plt.subplots(figsize=(6, 3))
    axes.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    axes.set_axis_off()
    figure.tight_layout()
    return figure


def _explanation(result: ShapResult):
    import shap

    return shap.Explanation(
        values=result.values,
        data=result.data,
        feature_names=list(result.feature_names),
        base_values=np.full(result.n_samples, float(result.base_value)),
    )


def _capture(draw, title: str):
    """Run a shap plotting call that draws on the current axes and hand back its Figure.

    shap's plotting helpers draw into pyplot state and return None, which is the opposite of this
    project's convention, so the figure is created here and reclaimed with gcf() afterwards.
    """
    figure = plt.figure()
    try:
        draw()
    except Exception as exc:
        plt.close(figure)
        return _message_figure(f"{title} could not be drawn: {type(exc).__name__}: {exc}")
    drawn = plt.gcf()
    if drawn is not figure:
        plt.close(figure)
    drawn.suptitle(title)
    drawn.tight_layout()
    return drawn


def shap_beeswarm(result: ShapResult, max_display: int = 25):
    """Per-feature beeswarm: one row per feature, one dot per sample, coloured by feature value."""
    import shap

    if result.n_samples == 0 or result.n_features == 0:
        return _message_figure("No SHAP values to plot")

    return _capture(
        lambda: shap.plots.beeswarm(_explanation(result), max_display=max_display, show=False),
        f"SHAP beeswarm - {result.target_name} ({result.output_space})",
    )


def shap_bar(result: ShapResult, max_display: int = 25):
    """Per-feature mean |SHAP| bar. shap folds the tail into a 'sum of N other features' row."""
    import shap

    if result.n_samples == 0 or result.n_features == 0:
        return _message_figure("No SHAP values to plot")

    return _capture(
        lambda: shap.plots.bar(_explanation(result), max_display=max_display, show=False),
        f"Mean |SHAP| - {result.target_name} ({result.output_space})",
    )


def shap_block_bar(result: ShapResult):
    """Rolled-up mean |SHAP| per block: static, categorical, auxiliary, and one bar per modality.

    The per-feature plots are capped at the top N, and with every band of every modality holding its
    own row the temporal branch can dominate in aggregate while no single band ranks highly enough to
    be displayed. This view is the antidote, and it is the direct quantitative answer to the question
    the soil_tabular ablation baseline exists to ask: what does the time series actually buy?
    """
    blocks = result.block_mean_abs()
    if not blocks:
        return _message_figure("No SHAP blocks to plot")

    names = list(blocks)
    heights = [blocks[name] for name in names]
    order = np.argsort(heights)

    figure, axes = plt.subplots(figsize=(7, max(2.5, 0.45 * len(names) + 1.2)))
    axes.barh([names[index] for index in order], [heights[index] for index in order], color="#4C78A8")
    axes.set_xlabel(f"mean |sum of SHAP within block| ({result.output_space})")
    axes.set_title(f"Contribution by block - {result.target_name}")
    axes.grid(axis="x", alpha=0.3)
    figure.tight_layout()
    return figure
