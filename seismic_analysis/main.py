"""
Pipeline completa di analisi delle storie temporali.

Fasi:
  1. Lettura di tutti i file nella cartella di input.
  2. Prima scrematura: esclusione delle componenti verticali (Z).
  3. Seconda scrematura: controllo di plausibilita' fisica sul dato
     grezzo (protezione da glitch del digitalizzatore, tipici dei dati
     non processati come ESM 'CV'): si scarta il file se il valore
     assoluto massimo del segnale grezzo supera max_plausible_accel_ms2.
  4. Terza scrematura, in base al tipo di dato:
     - dati gia' processati/filtrati (dichiarano HIGH_CUT_FREQUENCY_HZ
       nell'header): esclusi se il filtro originale era piu' stretto di
       min_original_high_cut_hz;
     - dati grezzi/non filtrati (campo assente, es. ESM 'CV'): esclusi
       solo se la loro frequenza di campionamento e' troppo bassa per
       applicare le bande di filtraggio richieste (Nyquist insufficiente).
  5. Per ogni file rimanente: filtraggio Butterworth passa-banda nelle due
     bande configurate, individuazione del superamento soglia e del picco
     di accelerazione assoluta (calcolato SEMPRE, anche senza superamento
     soglia), calcolo del delta t quando applicabile.
  6. Esportazione di un CSV con tutti i risultati (una riga per file x
     banda, con un campo `activated` che indica il superamento soglia),
     di un log delle esclusioni "strutturali" (Z, plausibilita' fisica,
     idoneita' al filtraggio) e di uno scatter plot riassuntivo (basato
     solo sulle storie attivate, dato che il delta_t ha senso solo in
     quel caso).
  7. Confronto tra le bande: per ogni storia, si verifica se l'attivazione
     e' concorde o discorde tra le due bande configurate (es. attivata a
     40Hz ma non a 20Hz), riportando in output anche il picco raggiunto
     in ENTRAMBE le bande (disponibile perche' calcolato sempre, vedi
     punto 5) — vedi band_comparison.csv/png e band_mismatch.csv.

Uso:
    python -m seismic_analysis.main [opzioni]

Eseguire con --help per l'elenco completo dei parametri configurabili.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from .analysis import analyze_filtered_signal
from .band_comparison import (
    build_band_comparison_table,
    compute_delay_by_pga_bins,
    compute_mismatch_by_pga_band,
    plot_band_comparison,
    plot_delay_by_pga_bins,
    plot_mismatch_by_pga_band,
)
from .config import Config, FilterBand
from .filtering import apply_bandpass
from .io_dyna import find_dyna_files, parse_dyna_file
from .plotting import plot_results
from .screening import check_filter_suitability, check_physical_plausibility, is_vertical_component


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=None, help="Cartella con i file da analizzare")
    parser.add_argument("--output-dir", default=None, help="Cartella di output")
    parser.add_argument("--order", type=int, default=None, help="Ordine dei filtri Butterworth")
    parser.add_argument("--threshold", type=float, default=None, help="Soglia di attivazione (valore assoluto, in m/s^2)")
    parser.add_argument("--min-high-cut", type=float, default=None,
                         help="Soglia minima di HIGH_CUT_FREQUENCY_HZ originale sotto la quale un file viene escluso")
    parser.add_argument("--max-plausible-accel", type=float, default=None,
                         help="Soglia [m/s^2] di plausibilita' fisica sul dato grezzo: file con valori "
                              "assoluti sopra questa soglia vengono scartati come probabili glitch "
                              "del digitalizzatore (default 50.0 m/s^2 ~ 5.1 g)")
    parser.add_argument("--band1", type=float, nargs=2, metavar=("LOW", "HIGH"), default=None,
                         help="Frequenze di taglio [Hz] della prima banda (default 0.1 20)")
    parser.add_argument("--band2", type=float, nargs=2, metavar=("LOW", "HIGH"), default=None,
                         help="Frequenze di taglio [Hz] della seconda banda (default 0.1 40)")
    parser.add_argument("--delta-t-percentile", type=float, default=None,
                         help="Percentile di delta_t per la prima linea verticale nel grafico "
                              "(default 5; usare -1 per disattivare questa linea)")
    parser.add_argument("--delta-t-fixed-threshold", type=float, default=None,
                         help="Tempo fisso [s] per la seconda linea verticale nel grafico: viene "
                              "mostrata la percentuale di storie con delta_t oltre questo valore "
                              "(default 1.0; usare -1 per disattivare questa linea)")
    parser.add_argument("--line-min-peak", type=float, default=None,
                         help="Seconda soglia [m/s^2] sull'accelerazione di picco: solo le storie "
                              "che la superano vengono usate per calcolare le due linee verticali "
                              "(default 0.0 = nessun filtro; non influenza lo scatter, che mostra "
                              "sempre tutti i punti)")
    parser.add_argument("--no-baseline-correction", action="store_true",
                         help="Disattiva la rimozione del trend lineare/taper prima del filtraggio "
                              "(per default e' attiva, consigliata sui dati grezzi non processati)")
    parser.add_argument("--taper-fraction", type=float, default=None,
                         help="Frazione di taper (Tukey) per lato applicata ai bordi prima del "
                              "filtraggio (default 0.05); 0 per disattivare solo il taper "
                              "mantenendo la rimozione del trend")
    parser.add_argument("--pga-bins", type=float, nargs="+", default=None,
                         help="Bordi delle fasce di PGA [m/s^2] per l'analisi dei ritardi medi e "
                              "delle discordanze stratificate (default: 0.05 0.5 1 2 4)")
    parser.add_argument("--x-axis-clip-percentile", type=float, default=None,
                         help="Percentile di delta_t (tutte le bande insieme) usato per il limite "
                              "destro dell'asse X nello scatter (default 95; alzarlo mostra piu' "
                              "coda a destra, utile per vedere quanto una banda sborda oltre l'altra)")
    parser.add_argument("--connect-bands", action="store_true",
                         help="Disegna una linea sottile che collega, nello scatter, il pallino di "
                              "una banda con quello dell'altra per la stessa storia (solo storie "
                              "attivate in tutte le bande). Disattivato di default: su popolazioni "
                              "grandi puo' affollare il grafico.")
    return parser


def config_from_args(args: argparse.Namespace) -> Config:
    config = Config()
    overrides = {}
    if args.input_dir is not None:
        overrides["input_dir"] = args.input_dir
    if args.output_dir is not None:
        overrides["output_dir"] = args.output_dir
    if args.order is not None:
        overrides["filter_order"] = args.order
    if args.threshold is not None:
        overrides["activation_threshold"] = args.threshold
    if args.min_high_cut is not None:
        overrides["min_original_high_cut_hz"] = args.min_high_cut
    if args.max_plausible_accel is not None:
        overrides["max_plausible_accel_ms2"] = args.max_plausible_accel
    if args.delta_t_percentile is not None:
        overrides["delta_t_percentile"] = args.delta_t_percentile
    if args.delta_t_fixed_threshold is not None:
        overrides["delta_t_fixed_threshold_s"] = args.delta_t_fixed_threshold
    if args.line_min_peak is not None:
        overrides["line_min_peak_accel_ms2"] = args.line_min_peak
    if args.no_baseline_correction:
        overrides["baseline_correction"] = False
    if args.taper_fraction is not None:
        overrides["taper_fraction"] = args.taper_fraction
    if args.pga_bins is not None:
        overrides["pga_bins_ms2"] = tuple(sorted(args.pga_bins))
    if args.x_axis_clip_percentile is not None:
        overrides["x_axis_clip_percentile"] = args.x_axis_clip_percentile
    if args.connect_bands:
        overrides["connect_band_points"] = True

    if args.band1 is not None or args.band2 is not None:
        bands = list(config.bands)
        if args.band1 is not None:
            bands[0] = replace(bands[0], low_hz=args.band1[0], high_hz=args.band1[1])
        if args.band2 is not None:
            bands[1] = replace(bands[1], low_hz=args.band2[0], high_hz=args.band2[1])
        overrides["bands"] = tuple(bands)

    return replace(config, **overrides)


def run(config: Config) -> pd.DataFrame:
    input_dir = Path(config.input_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = find_dyna_files(input_dir)
    if not files:
        raise FileNotFoundError(f"Nessun file trovato in {input_dir}")

    results_rows = []
    exclusion_rows = []
    n_candidate_files = 0  # file che superano entrambe le scremature

    for path in files:
        th = parse_dyna_file(path)

        # --- 1a scrematura: componente verticale (Z) esclusa a priori ---
        if is_vertical_component(th, config.excluded_component):
            exclusion_rows.append({
                "file": path.name,
                "motivo": f"componente {config.excluded_component} (verticale) esclusa a priori",
            })
            continue

        # --- 2a scrematura: plausibilita' fisica del dato grezzo ---
        ok, reason = check_physical_plausibility(th, config)
        if not ok:
            exclusion_rows.append({"file": path.name, "motivo": reason})
            continue

        # --- 3a scrematura: idoneita' del file al filtraggio richiesto ---
        max_band_high_hz = max(band.high_hz for band in config.bands)
        ok, reason = check_filter_suitability(th, config, max_band_high_hz)
        if not ok:
            exclusion_rows.append({"file": path.name, "motivo": reason})
            continue

        n_candidate_files += 1

        # --- analisi sulle due bande di filtraggio ---
        for band in config.bands:
            try:
                filtered = apply_bandpass(
                    th.data, band.low_hz, band.high_hz, th.fs, order=config.filter_order,
                    baseline_correction=config.baseline_correction,
                    taper_fraction=config.taper_fraction,
                )
            except ValueError as exc:
                exclusion_rows.append({
                    "file": path.name,
                    "motivo": f"filtraggio banda {band.name} non applicabile: {exc}",
                })
                continue

            result = analyze_filtered_signal(filtered, th.dt, config.activation_threshold)

            # Si salva SEMPRE una riga, anche se la soglia non e' stata
            # superata (result.activated=False): il picco viene comunque
            # calcolato e conservato, utile ad esempio per confrontare le
            # due bande anche nei casi di attivazione discorde (vedi
            # band_comparison.py). Le colonne t_cross_s/delta_t_s restano
            # vuote (None) quando activated=False, dato che il concetto
            # di "tempo dal superamento soglia" non si applica.
            results_rows.append({
                "file": path.name,
                "event_id": th.event_id,
                "component": th.stream,
                "band": band.name,
                "low_hz": band.low_hz,
                "high_hz": band.high_hz,
                "filter_order": config.filter_order,
                "threshold_ms2": config.activation_threshold,
                "activated": result.activated,
                "t_cross_s": result.t_cross_s,
                "t_peak_s": result.t_peak_s,
                "delta_t_s": result.delta_t_s,
                "peak_abs_accel_ms2": result.peak_value_abs,
                "peak_signed_accel_ms2": result.peak_value_signed,
            })

    results_df = pd.DataFrame(results_rows)
    exclusion_df = pd.DataFrame(exclusion_rows)

    # --- riepilogo per banda: quante storie hanno superato la soglia ---
    summary_rows = []
    for band in config.bands:
        if not results_df.empty:
            band_rows = results_df[results_df["band"] == band.name]
            n_exceeded = int(band_rows["activated"].sum())
        else:
            n_exceeded = 0
        n_not_exceeded = n_candidate_files - n_exceeded
        summary_rows.append({
            "band": band.name,
            "low_hz": band.low_hz,
            "high_hz": band.high_hz,
            "n_storie_candidate": n_candidate_files,
            "n_soglia_superata": n_exceeded,
            "n_soglia_non_superata": n_not_exceeded,
            "pct_soglia_superata": (100 * n_exceeded / n_candidate_files) if n_candidate_files else 0.0,
        })
    summary_df = pd.DataFrame(summary_rows)

    results_df.to_csv(output_dir / "results.csv", index=False)
    exclusion_df.to_csv(output_dir / "exclusion_log.csv", index=False)
    summary_df.to_csv(output_dir / "band_summary.csv", index=False)

    activated_df = results_df[results_df["activated"]] if not results_df.empty else results_df
    if not activated_df.empty:
        plot_results(results_df, config, output_dir / "scatter_delta_t_vs_peak.png", summary_df=summary_df)

    if not results_df.empty and len(config.bands) == 2:
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

    return results_df, summary_df


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = config_from_args(args)

    results_df, summary_df = run(config)

    print(f"File analizzati con successo: {results_df['file'].nunique() if not results_df.empty else 0}")
    print(f"Righe di risultato (file x banda): {len(results_df)}")
    print()
    print("Riepilogo superamento soglia per banda:")
    for _, row in summary_df.iterrows():
        print(
            f"  - {row['band']} ({row['low_hz']}-{row['high_hz']} Hz): "
            f"{int(row['n_soglia_superata'])}/{int(row['n_storie_candidate'])} storie "
            f"hanno superato la soglia ({row['pct_soglia_superata']:.1f}%)"
        )
    print()
    print(f"Risultati salvati in: {Path(config.output_dir) / 'results.csv'}")
    print(f"Log esclusioni salvato in: {Path(config.output_dir) / 'exclusion_log.csv'}")
    print(f"Riepilogo per banda salvato in: {Path(config.output_dir) / 'band_summary.csv'}")
    if not results_df.empty:
        activated_df = results_df[results_df["activated"]]
        if not activated_df.empty:
            print(f"Grafico salvato in: {Path(config.output_dir) / 'scatter_delta_t_vs_peak.png'}")
        if len(config.bands) == 2:
            print(f"Confronto tra bande salvato in: {Path(config.output_dir) / 'band_comparison.csv'}")
            print(f"Discordanze tra bande salvate in: {Path(config.output_dir) / 'band_mismatch.csv'}")
            print(f"Grafico di confronto salvato in: {Path(config.output_dir) / 'band_comparison.png'}")
            print(f"Ritardo medio per fascia di PGA salvato in: {Path(config.output_dir) / 'delay_by_pga_band.csv'}")
            print(f"Discordanze per fascia di PGA salvate in: {Path(config.output_dir) / 'mismatch_by_pga_band.csv'}")


if __name__ == "__main__":
    main()
