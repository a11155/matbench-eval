"""Analyze symmetry retention across models."""

# %%
import os

import numpy as np
import plotly.express as px
import pymatviz as pmv
from IPython.display import display
from plotly import graph_objects as go
from pymatviz.enums import Key
from pymatviz.utils import si_fmt

import matbench_discovery.metrics.geo_opt as go_metrics
from matbench_discovery import SITE_FIGS, today
from matbench_discovery.data import Model, df_wbm
from matbench_discovery.enums import MbdKey
from matbench_discovery.preds.geo_opt import df_sym

init_spg_col = "init_spg_num"
dft_spg_col = "dft_spg_num"
df_wbm[init_spg_col] = df_wbm[MbdKey.init_wyckoff].str.split("_").str[2].astype(int)
df_wbm[dft_spg_col] = df_wbm[Key.wyckoff_spglib].str.split("_").str[2].astype(int)
module_dir = os.path.dirname(__file__)

# limit the number of structures loaded per model to this number, 0 for no limit
debug_mode: int = 0

retained = (df_wbm[init_spg_col] == df_wbm[dft_spg_col]).sum()

print(
    f"DFT-relaxed structures that retained initial structure's spacegroup: "
    f"{retained:,} / {len(df_wbm):,} ({retained/len(df_wbm):.2%})"
)

models = df_sym.columns.levels[0]
print(f"\n{len(models)=}: {', '.join(models)}")


# %% Plot violin plot of RMSD vs DFT
fig_rmsd = px.violin(
    df_sym.xs(MbdKey.structure_rmsd_vs_dft, level=MbdKey.sym_prop, axis="columns")
    .round(3)
    .dropna(),
    orientation="h",
    color="model",
    box=True,
)
title = "RMSD of ML-relaxed structures vs DFT-relaxed structures"
fig_rmsd.layout.title = dict(text=title, x=0.5)
fig_rmsd.layout.xaxis.title = "Model"
fig_rmsd.layout.yaxis.title = "RMSD (Å)"
fig_rmsd.layout.margin.t = 50
fig_rmsd.update_traces(orientation="h", side="positive", width=1.8)

# add annotation for mean for each model
for model, srs_rmsd in df_sym.xs(
    MbdKey.structure_rmsd_vs_dft, level=MbdKey.sym_prop, axis="columns"
).items():
    mean_rmsd = srs_rmsd.mean()
    model_color = next(
        (trace["marker"]["color"] for trace in fig_rmsd.data if trace["name"] == model),
        "black",
    )
    fig_rmsd.add_annotation(
        x=mean_rmsd,
        y=model,
        text=f"Mean: {mean_rmsd:.3f}",
        ax=0,
        ay=-60,
        font=dict(size=12, color=model_color),
        arrowcolor=model_color,
    )

fig_rmsd.layout.height = len(fig_rmsd.data) * 100
fig_rmsd.layout.showlegend = False
fig_rmsd.show()

pmv.save_fig(fig_rmsd, f"{module_dir}/{today}-rmsd-violin.pdf")


# %% calculate number of model spacegroups agreeing with DFT-relaxed spacegroup
avg_spg_diff = (
    df_sym.xs(MbdKey.spg_num_diff, level=MbdKey.sym_prop, axis="columns")
    .mean(axis=0)
    .round(1)
)
print(f"Average spacegroup number difference vs DFT for each model: {avg_spg_diff}")


# %% violin plot of spacegroup number diff vs DFT
fig_sym = px.violin(
    df_sym.xs(MbdKey.spg_num_diff, level=MbdKey.sym_prop, axis="columns"),
    title="Spacegroup Number Diff vs DFT",
    orientation="h",
    color="model",
)
fig_sym.layout.xaxis.title = "Model"
fig_sym.layout.yaxis.title = "Spacegroup Number Diff vs DFT"
fig_sym.layout.margin.t = 50
fig_sym.layout.height = 600
fig_sym.update_traces(orientation="h", side="positive", width=1.8)

fig_sym.show()
pmv.save_fig(fig_sym, f"{module_dir}/{today}-sym-violin.pdf")


# %% violin plot of number of symmetry operations in ML-relaxed structures
fig_sym_ops = px.violin(
    df_sym.xs(Key.n_sym_ops, level=MbdKey.sym_prop, axis="columns"),
    title="Number of Symmetry Operations in ML-relaxed Structures",
    orientation="h",
    color="model",
)
fig_sym_ops.layout.xaxis.title = "Model"
fig_sym_ops.layout.yaxis.title = "Number of Symmetry Operations N<sub>sop</sub>"
fig_sym_ops.layout.margin.t = 50
fig_sym_ops.layout.height = 600
fig_sym_ops.update_traces(orientation="h", side="positive", width=1.8)
fig_sym_ops.show()
pmv.save_fig(fig_sym_ops, f"{module_dir}/{today}-sym-ops-violin.pdf")


# %% violin plot of number of symmetry operations in ML-relaxed structures vs DFT
fig_sym_ops_diff = px.violin(
    df_sym.drop(Key.dft.label, level=Key.model, axis="columns")
    .xs(MbdKey.n_sym_ops_diff, level=MbdKey.sym_prop, axis="columns")
    .reset_index(),
    orientation="h",
    color="model",
    hover_name=Key.mat_id,
    spanmode="hard",
)
title = "Difference in number of symmetry operations in ML vs DFT-relaxed structures"
fig_sym_ops_diff.layout.title = dict(text=title, x=0.5)
fig_sym_ops_diff.layout.xaxis.title = "N<sub>sop,ML</sub> - N<sub>sop,DFT</sub>"
fig_sym_ops_diff.layout.yaxis.title = "Model"
fig_sym_ops_diff.layout.margin.t = 50
fig_sym_ops_diff.update_traces(orientation="h", side="positive", width=1.8)

fig_sym_ops_diff.show()
pmv.save_fig(fig_sym_ops_diff, f"{module_dir}/{today}-sym-ops-diff-violin.svelte")


# %% Print summary of symmetry changes
df_sym_changes = go_metrics.analyze_symmetry_changes(df_sym).convert_dtypes()
display(
    df_sym_changes.round(3)
    .rename(columns=lambda col: col.removeprefix("symmetry_"))
    .style.format(lambda x: f"{x:.1%}" if isinstance(x, float) else si_fmt(x))
    .background_gradient(cmap="Oranges", subset="decrease")
    .background_gradient(cmap="Greens", subset="match")
    .background_gradient(cmap="Blues", subset="increase")
    .set_caption("Symmetry changes vs DFT")
)


# %% Plot cumulative distribution of RMSD vs DFT
fig_rmsd_cdf = go.Figure()
x_max = 0.05

models = df_sym_changes.index

# Calculate and plot CDF for each model
for model in models:
    rmsd_vals = df_sym.xs(
        MbdKey.structure_rmsd_vs_dft, level=MbdKey.sym_prop, axis="columns"
    )[model].dropna()

    # Calculate CDF
    sorted_rmsds = np.sort(rmsd_vals)
    cumulative = np.arange(1, len(sorted_rmsds) + 1) / len(sorted_rmsds)

    # Sample 200 points evenly across the range
    n_points = 200
    indices = np.linspace(0, len(sorted_rmsds) - 1, n_points, dtype=int)
    sampled_rmsds = sorted_rmsds[indices]
    sampled_cumulative = cumulative[indices]

    # calculate AUC only up to x_max
    AUC = (
        np.trapezoid(
            sampled_cumulative[sampled_rmsds <= x_max],
            sampled_rmsds[sampled_rmsds <= x_max],
        )
    ) / x_max
    # Add CDF line for this model
    fig_rmsd_cdf.add_scatter(
        x=sampled_rmsds,
        y=sampled_cumulative,
        name=f"{model} · {AUC=:.3}",
        mode="lines",
        hovertemplate=f"<b>{model}</b><br>RMSD: %{{x:.3f}} Å<br>Cumulative: %{{y:.1%}}"
        "<extra></extra>",
    )

# sort by AUC
fig_rmsd_cdf.data = sorted(
    fig_rmsd_cdf.data, key=lambda trace: -float(trace["name"].split("AUC=")[1])
)

fig_rmsd_cdf.layout.xaxis.update(title="RMSD (Å)", range=[0, x_max])
fig_rmsd_cdf.layout.yaxis.update(title="Cumulative", tickformat=".0%", range=[0, 1])
fig_rmsd_cdf.layout.legend = dict(y=0, xanchor="right", x=1)

pmv.save_fig(fig_rmsd_cdf, f"{SITE_FIGS}/struct-rmsd-cdf-models.svelte")

title = "Cumulative Distribution of RMSD vs DFT-relaxed structures"
fig_rmsd_cdf.layout.title = dict(text=title, x=0.5)
fig_rmsd_cdf.layout.margin.t = 40
fig_rmsd_cdf.show()


# %%
if __name__ == "__main__":
    go_metrics.write_geo_opt_metrics_to_yaml(df_sym, df_sym_changes)

    # %% plot ML vs DFT relaxed spacegroup correspondence as sankey diagrams
    df_spg = df_sym.xs(Key.spg_num, level=MbdKey.sym_prop, axis="columns")
    for model in {*df_spg} - {Key.dft}:
        # get most common pairs of DFT/Model spacegroups
        common_dft_spgs, common_model_spgs = zip(
            *df_spg[[Key.dft, model]].value_counts().head(10).index
        )
        model_label = getattr(Model, model).label
        df_spg_common = (
            df_spg.query(f"dft in {common_dft_spgs} and {model} in {common_model_spgs}")
            .sort_values(by=Key.dft)
            .rename(columns={model: model_label, Key.dft: Key.dft.label})
        )

        fig = pmv.sankey_from_2_df_cols(
            df_spg_common.reset_index(),
            [Key.dft.label, model_label],
            label_standoff=35,
        )
        fig.show()
        pmv.save_fig(fig, f"{SITE_FIGS}/spg-sankey-{model.replace('_', '-')}.svelte")


# TODO maybe add average_distance_within_threshold metric
# https://github.com/FAIR-Chem/fairchem/blob/6329e922/src/fairchem/core/modules/evaluator.py#L318
