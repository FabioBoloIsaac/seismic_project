"""
Analisi delle discordanze di attivazione tra bande (20Hz vs 40Hz),
usando SOLO i CSV gia' prodotti da main.py (results.csv + exclusion_log.csv)
— senza rileggere/rifiltrare le storie temporali originali.

Per ogni storia temporale, si determina se ha attivato (superato la
soglia) in ciascuna banda. Una "discordanza" e' il caso in cui ha
attivato in una banda ma non nell'altra.

Oltre al conteggio/dettaglio delle discordanze, produce anche il ritardo
medio (delta_t) per fascia di PGA e le discordanze stratificate per
fascia di PGA — le stesse analisi disponibili in automatico in main.py
(vedi band_comparison.py), qui ricalcolate dai soli CSV.

LIMITE IMPORTANTE: da questi due CSV e' possibile recuperare il picco di
accelerazione SOLO per la banda che ha attivato (e' l'unico dato salvato
in results.csv). Il picco della banda che NON ha attivato non e' mai
stato calcolato/salvato da main.py quando la soglia non viene superata:
exclusion_log.csv registra solo il motivo testuale dell'esclusione, non
il valore di accelerazione raggiunto. Per avere anche quel dato, la
pipeline principale (main.py) e' stata aggiornata per salvarlo sempre:
rieseguendo l'analisi con la versione aggiornata, lo stesso confronto e'
disponibile automaticamente e con l'informazione completa (vedi
band_comparison.csv/png prodotti da main.py).

Uso:
    python -m seismic_analysis.band_mismatch_from_csv \
        --results-csv results.csv --exclusion-csv exclusion_log.csv \
        --output-dir output --pga-bins 0.05 0.5 1 2 4
"""

from __future__ import annotations

import argparse
import itertools
import re
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .band_comparison import (
    build_band_comparison_table,
    compute_delay_by_pga_bins,
    compute_mismatch_by_pga_band,
    plot_delay_by_pga_bins,
    plot_mismatch_by_pga_band,
)
from .config import Config, FilterBand

# Pattern del motivo di esclusione per soglia non superata, come generato
# da main.py: "soglia di attivazione (0.05) mai superata nella banda 20Hz"
_THRESHOLD_MOTIVE_RE = re.compile(r"soglia di attivazione .* mai superata nella banda (\S+)")

_FALLBACK_COLORS = ["tab:blue", "tab:red", "tab:green", "tab:purple", "tab:orange", "tab:brown"]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-csv", required=True, help="Percorso di results.csv")
    parser.add_argument("--exclusion-csv", required=True, help="Percorso di exclusion_log.csv")
    parser.add_argument("--output-dir", default="output", help="Cartella di output")
    parser.add_argument("--pga-bins", type=float, nargs="+", default=None,
                         help="Bordi delle fasce di PGA [m/s^2] (default: 0.05 0.5 1 2 4)")
    parser.add_argument("--activation-threshold", type=float, default=None,
                         help="Soglia di attivazione originale (default 0.05), solo per riferimento nei grafici")
    return parser


def load_activation_table(results_csv: Path, exclusion_csv: Path) -> pd.DataFrame:
    """Costruisce una tabella file x banda con lo stato di attivazione e,
    quando disponibile, il picco/delta_t raggiunto."""

    results_df = pd.read_csv(results_csv)
    exclusion_df = pd.read_csv(exclusion_csv)

    activated_rows = results_df[["file", "band", "event_id", "component",
                                  "peak_abs_accel_ms2", "delta_t_s"]].copy()
    activated_rows["activated"] = True

    # Isola solo le esclusioni per soglia non superata (ignora Z, filtro,
    # plausibilita' fisica: quelle non sono specifiche di una banda).
    matches = exclusion_df["motivo"].str.extract(_THRESHOLD_MOTIVE_RE)
    threshold_excl = exclusion_df.assign(band=matches[0])
    threshold_excl = threshold_excl[threshold_excl["band"].notna()][["file", "band"]].copy()
    threshold_excl["activated"] = False
    threshold_excl["peak_abs_accel_ms2"] = float("nan")
    threshold_excl["delta_t_s"] = float("nan")

    combined = pd.concat([activated_rows, threshold_excl], ignore_index=True)
    # event_id/component non sono in exclusion_log: li recupera da results_df se disponibili
    lookup = results_df.drop_duplicates("file")[["file", "event_id", "component"]]
    combined = combined.drop(columns=["event_id", "component"], errors="ignore").merge(lookup, on="file", how="left")
    return combined


def bands_from_activation_table(activation_table: pd.DataFrame) -> tuple[FilterBand, ...]:
    """Ricostruisce oggetti FilterBand (solo nome e colore, servono per
    riusare le funzioni di band_comparison.py) dai nomi banda presenti
    nella tabella, riusando i colori di default quando il nome coincide."""

    default_colors = {b.name: b.color for b in Config().bands}
    fallback_cycle = itertools.cycle(_FALLBACK_COLORS)
    bands = []
    for name in sorted(activation_table["band"].unique()):
        color = default_colors.get(name) or next(fallback_cycle)
        bands.append(FilterBand(name=name, low_hz=0.0, high_hz=0.0, color=color))
    return tuple(bands)


def plot_mismatches(mismatch: pd.DataFrame, band_a: str, band_b: str, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    counts = mismatch["categoria"].value_counts()
    axes[0].bar(counts.index, counts.values, color=["tab:blue", "tab:red"] if len(counts) <= 2 else None)
    axes[0].set_ylabel("Numero di storie discordanti")
    axes[0].set_title("Conteggio discordanze tra bande")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v, str(v), ha="center", va="bottom")

    for tipo, color in [(f"solo {band_a}", "tab:blue"), (f"solo {band_b}", "tab:red")]:
        subset = mismatch[mismatch["categoria"] == tipo]
        peak_col = f"picco_{band_a}_ms2" if tipo == f"solo {band_a}" else f"picco_{band_b}_ms2"
        peaks = subset[peak_col].dropna()
        if not peaks.empty:
            axes[1].hist(peaks, bins=30, alpha=0.6, label=tipo, color=color)

    axes[1].set_xlabel("Picco raggiunto nella banda che ha attivato [m/s²]")
    axes[1].set_ylabel("Numero di storie")
    axes[1].set_title("Distribuzione del picco nelle storie discordanti\n(picco della sola banda attivata — l'altra non è disponibile da questi CSV)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    all_peaks = pd.concat([
        mismatch[f"picco_{band_a}_ms2"].dropna(),
        mismatch[f"picco_{band_b}_ms2"].dropna(),
    ])
    if not all_peaks.empty:
        margin = (all_peaks.max() - all_peaks.min()) * 0.1 or all_peaks.max() * 0.1 or 0.01
        axes[1].set_xlim(max(0, all_peaks.min() - margin), all_peaks.max() + margin)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = Config()
    if args.pga_bins is not None:
        config = replace(config, pga_bins_ms2=tuple(sorted(args.pga_bins)))
    if args.activation_threshold is not None:
        config = replace(config, activation_threshold=args.activation_threshold)

    activation_table = load_activation_table(Path(args.results_csv), Path(args.exclusion_csv))
    bands = bands_from_activation_table(activation_table)
    band_a, band_b = bands[0].name, bands[1].name

    comparison_table = build_band_comparison_table(activation_table, bands)
    mismatch = comparison_table[comparison_table["discordanza"]]

    mismatch_path = output_dir / "band_mismatch_from_csv.csv"
    chart_path = output_dir / "band_mismatch_from_csv.png"
    mismatch.to_csv(mismatch_path, index=False)
    plot_mismatches(mismatch, band_a, band_b, chart_path)

    delay_table = compute_delay_by_pga_bins(activation_table, config.pga_bins_ms2)
    delay_path = output_dir / "delay_by_pga_band_from_csv.csv"
    delay_chart_path = output_dir / "delay_by_pga_band_from_csv.png"
    delay_table.to_csv(delay_path, index=False)
    plot_delay_by_pga_bins(delay_table, replace(config, bands=bands), delay_chart_path)

    mismatch_by_threshold = compute_mismatch_by_pga_band(comparison_table, config.pga_bins_ms2, band_a, band_b)
    threshold_path = output_dir / "mismatch_by_pga_band_from_csv.csv"
    threshold_chart_path = output_dir / "mismatch_by_pga_band_from_csv.png"
    mismatch_by_threshold.to_csv(threshold_path, index=False)
    plot_mismatch_by_pga_band(mismatch_by_threshold, config, threshold_chart_path)

    n_total_files = activation_table["file"].nunique()
    print(f"File analizzati: {n_total_files}")
    print(f"Discordanze trovate: {len(mismatch)}")
    print(mismatch["categoria"].value_counts().to_string())
    print()
    print(f"Dettaglio discordanze: {mismatch_path}")
    print(f"Grafico discordanze: {chart_path}")
    print(f"Ritardo medio per fascia di PGA: {delay_path}")
    print(f"Discordanze per fascia di PGA: {threshold_path}")
    print()
    print("NOTA: il picco riportato per le storie discordanti è solo quello")
    print("della banda che ha attivato. Per avere anche il picco della banda")
    print("che non ha attivato, riesegui main.py (versione aggiornata) e usa")
    print("band_comparison.csv/png in output/.")


if __name__ == "__main__":
    main()
