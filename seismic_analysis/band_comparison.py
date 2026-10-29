"""
Confronto dell'attivazione tra le due bande di filtraggio.

A differenza di band_mismatch_from_csv.py (che lavora sui CSV gia'
prodotti e ha accesso al picco solo per la banda che ha attivato), questo
modulo lavora direttamente sul DataFrame `results_df` prodotto da
main.py, che ora contiene SEMPRE il picco (anche per le storie non
attivate): il confronto e' quindi completo, con il picco di entrambe le
bande disponibile in ogni caso, comprese le discordanze.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import Config, FilterBand


def build_band_comparison_table(results_df: pd.DataFrame, bands: Sequence[FilterBand]) -> pd.DataFrame:
    """Pivotta results_df (una riga per file x banda) in una riga per
    file, con attivazione e picco di entrambe le bande affiancati, piu'
    un flag `discordanza` (True se l'attivazione differisce tra le due)."""

    band_names = [b.name for b in bands]
    if len(band_names) != 2:
        raise ValueError(f"Il confronto richiede esattamente 2 bande, trovate: {band_names}")
    band_a, band_b = band_names

    pivot_activated = results_df.pivot_table(index="file", columns="band", values="activated", aggfunc="first")
    pivot_peak = results_df.pivot_table(index="file", columns="band", values="peak_abs_accel_ms2", aggfunc="first")
    pivot_delta_t = results_df.pivot_table(index="file", columns="band", values="delta_t_s", aggfunc="first")
    lookup = results_df.drop_duplicates("file").set_index("file")[["event_id", "component"]]

    table = pd.DataFrame({
        "event_id": lookup["event_id"],
        "component": lookup["component"],
        f"attivata_{band_a}": pivot_activated.get(band_a),
        f"attivata_{band_b}": pivot_activated.get(band_b),
        f"picco_{band_a}_ms2": pivot_peak.get(band_a),
        f"picco_{band_b}_ms2": pivot_peak.get(band_b),
        f"delta_t_{band_a}_s": pivot_delta_t.get(band_a),
        f"delta_t_{band_b}_s": pivot_delta_t.get(band_b),
    }).reset_index()

    table["discordanza"] = table[f"attivata_{band_a}"] != table[f"attivata_{band_b}"]

    def _categoria(row):
        a, b = row[f"attivata_{band_a}"], row[f"attivata_{band_b}"]
        if a and b:
            return "entrambe"
        if a and not b:
            return f"solo {band_a}"
        if b and not a:
            return f"solo {band_b}"
        return "nessuna"

    table["categoria"] = table.apply(_categoria, axis=1)
    return table


def _pga_bin_labels(edges: Sequence[float]) -> list[str]:
    labels = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        labels.append(f">= {lo:g}" if np.isinf(hi) else f"[{lo:g}, {hi:g})")
    return labels


def compute_delay_by_pga_bins(long_df: pd.DataFrame, bins: Sequence[float]) -> pd.DataFrame:
    """Ritardo medio (delta_t) per banda, per fascia di picco (PGA).

    long_df: DataFrame "lungo" (una riga per file x banda) con almeno le
    colonne band, peak_abs_accel_ms2, delta_t_s — opzionalmente
    `activated` (se assente, si assume che ogni riga sia gia' una storia
    attivata, come nei CSV di versioni precedenti). Il ritardo ha senso
    solo per le storie attivate: quelle non attivate vengono escluse qui.
    """

    activated = long_df[long_df["activated"]] if "activated" in long_df.columns else long_df
    activated = activated.dropna(subset=["delta_t_s", "peak_abs_accel_ms2"])
    if activated.empty:
        return pd.DataFrame(columns=["band", "fascia_pga", "n_storie", "delta_t_medio_s", "delta_t_mediano_s", "delta_t_std_s"])

    edges = list(bins) + [np.inf]
    labels = _pga_bin_labels(edges)
    activated = activated.copy()
    activated["fascia_pga"] = pd.cut(activated["peak_abs_accel_ms2"], bins=edges, labels=labels,
                                      right=False, include_lowest=True)

    grouped = activated.groupby(["band", "fascia_pga"], observed=False).agg(
        n_storie=("delta_t_s", "count"),
        delta_t_medio_s=("delta_t_s", "mean"),
        delta_t_mediano_s=("delta_t_s", "median"),
        delta_t_std_s=("delta_t_s", "std"),
    ).reset_index()
    return grouped


def compute_mismatch_by_pga_band(wide_table: pd.DataFrame, bins: Sequence[float],
                                  band_a: str, band_b: str) -> pd.DataFrame:
    """Per ciascuna fascia di PGA (bordi NON sovrapposti, come in
    compute_delay_by_pga_bins: [bordo[i], bordo[i+1]), l'ultima aperta
    verso l'alto), quante storie sono discordanti tra le bande, in totale
    e per tipo. Ogni storia conta in UNA sola fascia (quella del suo
    picco massimo tra le due bande), mai in piu' di una."""

    peak_a_col = f"picco_{band_a}_ms2"
    peak_b_col = f"picco_{band_b}_ms2"
    pga_proxy = wide_table[[peak_a_col, peak_b_col]].max(axis=1, skipna=True)

    edges = list(bins) + [np.inf]
    labels = _pga_bin_labels(edges)
    fascia = pd.cut(pga_proxy, bins=edges, labels=labels, right=False, include_lowest=True)

    rows = []
    for label in labels:
        subset = wide_table[fascia == label]
        n_tot = len(subset)
        n_mismatch = int(subset["discordanza"].sum()) if n_tot else 0
        n_only_a = int((subset["categoria"] == f"solo {band_a}").sum()) if n_tot else 0
        n_only_b = int((subset["categoria"] == f"solo {band_b}").sum()) if n_tot else 0
        rows.append({
            "fascia_pga": label,
            "n_storie_in_fascia": n_tot,
            "n_discordanti": n_mismatch,
            "pct_discordanti": (100 * n_mismatch / n_tot) if n_tot else 0.0,
            f"n_solo_{band_a}": n_only_a,
            f"n_solo_{band_b}": n_only_b,
        })
    return pd.DataFrame(rows)


def plot_delay_by_pga_bins(delay_table: pd.DataFrame, config: Config, output_path: Path) -> None:
    if delay_table.empty or delay_table["n_storie"].sum() == 0:
        return

    band_colors = {b.name: b.color for b in config.bands}
    edges = list(config.pga_bins_ms2) + [np.inf]
    bin_labels = _pga_bin_labels(edges)
    band_names = [b.name for b in config.bands]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(bin_labels))
    width = 0.8 / max(len(band_names), 1)

    for i, band_name in enumerate(band_names):
        sub = delay_table[delay_table["band"] == band_name].set_index("fascia_pga").reindex(bin_labels)
        offset = (i - (len(band_names) - 1) / 2) * width
        bars = ax.bar(x + offset, sub["delta_t_medio_s"], width=width,
                       yerr=sub["delta_t_std_s"], capsize=3,
                       color=band_colors.get(band_name, None), label=band_name, alpha=0.85)
        for xi, n, val in zip(x + offset, sub["n_storie"], sub["delta_t_medio_s"]):
            if pd.notna(val):
                ax.text(xi, val, f"n={int(n)}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=20, ha="right")
    ax.set_xlabel("Fascia di picco (PGA) [m/s²]")
    ax.set_ylabel("Ritardo medio (delta_t) [s]")
    ax.set_title("Ritardo medio tra superamento soglia e picco, per fascia di PGA")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_mismatch_by_pga_band(mismatch_table: pd.DataFrame, config: Config, output_path: Path) -> None:
    if mismatch_table.empty:
        return

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.bar(range(len(mismatch_table)), mismatch_table["n_storie_in_fascia"],
            color="lightgray", alpha=0.7, label="Storie nella fascia (asse sx)")
    ax1.set_ylabel("Numero di storie nella fascia")
    ax1.set_xticks(range(len(mismatch_table)))
    ax1.set_xticklabels(mismatch_table["fascia_pga"], rotation=20, ha="right")
    ax1.set_xlabel("Fascia di PGA [m/s²]")

    ax2 = ax1.twinx()
    ax2.plot(range(len(mismatch_table)), mismatch_table["pct_discordanti"],
              color="tab:red", marker="o", linewidth=2, label="% discordanti (asse dx)")
    ax2.set_ylabel("Percentuale di storie discordanti [%]", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.set_ylim(bottom=0)

    for i, row in mismatch_table.iterrows():
        ax2.annotate(f"{row['n_discordanti']}", (i, row["pct_discordanti"]),
                     textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, color="tab:red")

    ax1.set_title("Percentuale di storie discordanti tra bande, per fascia di PGA\n(fasce non sovrapposte: ogni storia conta in una sola fascia)")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")
    ax1.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_band_comparison(table: pd.DataFrame, config: Config, output_path: Path) -> None:
    bands = config.bands
    band_a, band_b = bands[0].name, bands[1].name
    color_a, color_b = bands[0].color, bands[1].color

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Pannello 1: scatter picco_a vs picco_b, colorato per categoria ---
    cat_colors = {
        "entrambe": "tab:gray",
        f"solo {band_a}": color_a,
        f"solo {band_b}": color_b,
        "nessuna": "lightgray",
    }
    for categoria, color in cat_colors.items():
        subset = table[table["categoria"] == categoria]
        if subset.empty:
            continue
        alpha = 0.8 if categoria != "entrambe" and categoria != "nessuna" else 0.25
        axes[0].scatter(
            subset[f"picco_{band_a}_ms2"], subset[f"picco_{band_b}_ms2"],
            s=20, alpha=alpha, color=color, label=f"{categoria} (n={len(subset)})",
            edgecolors="none",
        )

    axes[0].axvline(config.activation_threshold, color="k", linestyle=":", linewidth=1, alpha=0.6)
    axes[0].axhline(config.activation_threshold, color="k", linestyle=":", linewidth=1, alpha=0.6)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel(f"Picco banda {band_a} [m/s²]")
    axes[0].set_ylabel(f"Picco banda {band_b} [m/s²]")
    axes[0].set_title("Picco raggiunto nelle due bande (scala log)\nlinee tratteggiate = soglia di attivazione")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3, which="both")

    # --- Pannello 2: distribuzione dei picchi nelle sole storie discordanti ---
    mismatch = table[table["discordanza"]]
    only_a = mismatch[mismatch["categoria"] == f"solo {band_a}"]
    only_b = mismatch[mismatch["categoria"] == f"solo {band_b}"]

    # per ciascuna storia discordante, picco della banda attivata e
    # picco della banda NON attivata (ora sempre disponibile)
    peaks_activated = pd.concat([only_a[f"picco_{band_a}_ms2"], only_b[f"picco_{band_b}_ms2"]])
    peaks_not_activated = pd.concat([only_a[f"picco_{band_b}_ms2"], only_b[f"picco_{band_a}_ms2"]])

    if not mismatch.empty:
        bins = np.linspace(
            min(peaks_activated.min(), peaks_not_activated.min()),
            max(peaks_activated.max(), peaks_not_activated.max()),
            30,
        )
        axes[1].hist(peaks_activated, bins=bins, alpha=0.6, label="Picco nella banda ATTIVATA", color="tab:green")
        axes[1].hist(peaks_not_activated, bins=bins, alpha=0.6, label="Picco nella banda NON attivata", color="tab:orange")
        axes[1].axvline(config.activation_threshold, color="k", linestyle=":", linewidth=1.3, label="Soglia di attivazione")
        axes[1].legend(fontsize=8)
    else:
        axes[1].text(0.5, 0.5, "Nessuna storia discordante tra le due bande",
                      ha="center", va="center", transform=axes[1].transAxes, fontsize=10, color="gray")

    axes[1].set_xlabel("Accelerazione di picco [m/s²]")
    axes[1].set_ylabel("Numero di storie")
    axes[1].set_title(f"Storie discordanti (n={len(mismatch)}):\npicco nella banda attivata vs. nella banda non attivata")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
