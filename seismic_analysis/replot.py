"""
Rigenera il grafico scatter (delta_t vs picco) e le analisi di confronto
tra bande a partire da un file results.csv GIA' PRODOTTO da main.py,
senza rileggere/rifiltrare le storie temporali originali.

Utile per esplorare rapidamente l'effetto dei parametri che riguardano
SOLO il grafico (percentile, tempo fisso, seconda soglia sulle linee,
fasce di PGA, percentile dell'asse X, ecc.): si esegue main.py una volta
sola, poi si rilancia questo script tutte le volte che serve, cambiando
solo questi parametri — in pochi secondi anche su decine di migliaia di
righe, perche' non rifiltra nulla.

Se results.csv contiene la colonna `activated` (prodotta dalle versioni
recenti di main.py, con una riga per ogni file x banda anche senza
superamento soglia) vengono rigenerati anche:
  - il confronto tra bande (band_comparison.csv/png, band_mismatch.csv),
  - il ritardo medio per fascia di PGA (delay_by_pga_band.csv/png),
  - le discordanze per fascia di PGA (mismatch_by_pga_band.csv/png).
Su CSV di versioni precedenti (senza quella colonna) queste tre analisi
richiedono band_mismatch_from_csv.py invece di questo script.

IMPORTANTE: i parametri che riguardano il FILTRAGGIO delle storie
temporali (ordine del filtro, bande di frequenza, soglia di attivazione,
preprocessing, scremature, ecc.) NON possono essere cambiati qui: sono
gia' "congelati" nei valori di results.csv. Per cambiare quelli serve
rieseguire main.py da capo sui dati grezzi.

Uso:
    python -m seismic_analysis.replot --results-csv output/results.csv \
        --output-dir output_rev2 \
        --delta-t-percentile 10 \
        --line-min-peak 0.3 \
        --pga-bins 0.05 0.5 1 2 4

Eseguire con --help per l'elenco completo dei parametri.
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import replace
from pathlib import Path

import pandas as pd

from .band_comparison import (
    build_band_comparison_table,
    compute_delay_by_pga_bins,
    compute_mismatch_by_pga_band,
    plot_band_comparison,
    plot_delay_by_pga_bins,
    plot_mismatch_by_pga_band,
)
from .config import Config, FilterBand
from .plotting import plot_results

# Palette di riserva per bande il cui nome non coincide con quelle di
# default in Config (20Hz/40Hz): garantisce comunque colori distinti e
# stabili (stesso ordine ad ogni esecuzione, dato che i nomi banda in
# results.csv sono sempre gli stessi per uno stesso file).
_FALLBACK_COLORS = ["tab:blue", "tab:red", "tab:green", "tab:purple", "tab:orange", "tab:brown"]


def bands_from_results(results_df: pd.DataFrame) -> tuple[FilterBand, ...]:
    """Ricostruisce le FilterBand (nome, frequenze, colore) direttamente
    dalle colonne band/low_hz/high_hz di results.csv, cosi' questo script
    funziona anche se le bande non sono le 20Hz/40Hz di default. Se un
    nome banda coincide con una di Config() di default, ne riusa il
    colore per coerenza visiva con il grafico originale."""

    default_colors = {b.name: b.color for b in Config().bands}
    fallback_cycle = itertools.cycle(_FALLBACK_COLORS)

    bands = []
    for name in results_df["band"].drop_duplicates():
        row = results_df[results_df["band"] == name].iloc[0]
        color = default_colors.get(name) or next(fallback_cycle)
        bands.append(FilterBand(name=name, low_hz=row["low_hz"], high_hz=row["high_hz"], color=color))
    return tuple(bands)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-csv", required=True, help="Percorso di un results.csv gia' prodotto da main.py")
    parser.add_argument("--output", default=None,
                         help="Percorso dello scatter PNG da generare (default: accanto al CSV, "
                              "'<nome_csv>_replot.png')")
    parser.add_argument("--output-dir", default=None,
                         help="Cartella per gli output aggiuntivi (confronto bande, fasce PGA); "
                              "default: la cartella di --output, o del CSV se --output non e' dato")
    parser.add_argument("--delta-t-percentile", type=float, default=None,
                         help="Percentile di delta_t per la prima linea verticale (default 5; -1 per disattivare)")
    parser.add_argument("--delta-t-fixed-threshold", type=float, default=None,
                         help="Tempo fisso [s] per la seconda linea verticale (default 1.0; -1 per disattivare)")
    parser.add_argument("--line-min-peak", type=float, default=None,
                         help="Seconda soglia [m/s^2] sul picco: filtra solo le storie usate per le linee "
                              "(default 0.0 = nessun filtro; non tocca lo scatter)")
    parser.add_argument("--pga-bins", type=float, nargs="+", default=None,
                         help="Bordi delle fasce di PGA [m/s^2] (default: 0.05 0.5 1 2 4)")
    parser.add_argument("--x-axis-clip-percentile", type=float, default=None,
                         help="Percentile di delta_t usato per il limite destro dell'asse X (default 95)")
    parser.add_argument("--connect-bands", action="store_true",
                         help="Disegna una linea sottile che collega il pallino di una banda con "
                              "quello dell'altra per la stessa storia (solo storie attivate in "
                              "tutte le bande). Disattivato di default.")
    return parser


def config_from_args(args: argparse.Namespace) -> Config:
    config = Config()
    overrides = {}
    if args.delta_t_percentile is not None:
        overrides["delta_t_percentile"] = args.delta_t_percentile
    if args.delta_t_fixed_threshold is not None:
        overrides["delta_t_fixed_threshold_s"] = args.delta_t_fixed_threshold
    if args.line_min_peak is not None:
        overrides["line_min_peak_accel_ms2"] = args.line_min_peak
    if args.pga_bins is not None:
        overrides["pga_bins_ms2"] = tuple(sorted(args.pga_bins))
    if args.x_axis_clip_percentile is not None:
        overrides["x_axis_clip_percentile"] = args.x_axis_clip_percentile
    if args.connect_bands:
        overrides["connect_band_points"] = True
    return replace(config, **overrides)


def main() -> None:
    args = build_arg_parser().parse_args()
    config = config_from_args(args)

    results_path = Path(args.results_csv)
    results_df = pd.read_csv(results_path)

    required_cols = {"band", "low_hz", "high_hz", "delta_t_s", "peak_abs_accel_ms2"}
    missing = required_cols - set(results_df.columns)
    if missing:
        raise ValueError(
            f"Il file {results_path} non sembra un results.csv valido: "
            f"mancano le colonne {sorted(missing)}"
        )

    # NOTA: si passa results_df COMPLETO (non filtrato su activated) a
    # plot_results, che ora gestisce internamente sia il filtro sulle
    # storie attivate sia l'individuazione delle "orfane" (attivate in
    # una banda ma non nell'altra, colorate in nero) — richiede pero' la
    # colonna 'activated' per farlo correttamente; sui CSV di versioni
    # precedenti (senza quella colonna) il comportamento resta quello
    # storico (tutte le righe sono gia' storie attivate).
    config = replace(config, bands=bands_from_results(results_df))

    output_path = Path(args.output) if args.output else results_path.with_name(results_path.stem + "_replot.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_results(results_df, config, output_path)

    print(f"Grafico rigenerato da {results_path} ({len(results_df)} righe)")
    print(f"Salvato in: {output_path}")

    if "activated" not in results_df.columns:
        print()
        print("NOTA: questo CSV non ha la colonna 'activated' (versione precedente di")
        print("main.py): confronto tra bande, ritardo per fascia di PGA e discordanze")
        print("per fascia non sono rigenerabili da qui. Usa band_mismatch_from_csv.py.")
        return

    if len(config.bands) != 2:
        return

    output_dir = Path(args.output_dir) if args.output_dir else output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison_table = build_band_comparison_table(results_df, config.bands)
    comparison_table.to_csv(output_dir / "band_comparison.csv", index=False)
    mismatch = comparison_table[comparison_table["discordanza"]]
    mismatch.to_csv(output_dir / "band_mismatch.csv", index=False)
    plot_band_comparison(comparison_table, config, output_dir / "band_comparison.png")

    band_a, band_b = config.bands[0].name, config.bands[1].name

    delay_table = compute_delay_by_pga_bins(results_df, config.pga_bins_ms2)
    delay_table.to_csv(output_dir / "delay_by_pga_band.csv", index=False)
    plot_delay_by_pga_bins(delay_table, config, output_dir / "delay_by_pga_band.png")

    mismatch_by_threshold = compute_mismatch_by_pga_band(comparison_table, config.pga_bins_ms2, band_a, band_b)
    mismatch_by_threshold.to_csv(output_dir / "mismatch_by_pga_band.csv", index=False)
    plot_mismatch_by_pga_band(mismatch_by_threshold, config, output_dir / "mismatch_by_pga_band.png")

    print(f"Confronto tra bande rigenerato in: {output_dir}")


if __name__ == "__main__":
    main()
