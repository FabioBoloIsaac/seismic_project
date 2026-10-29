"""
Calibrazione statistica della soglia di attivazione.

A differenza di seismic_analysis/main.py (che parte da una soglia
FISSATA e ne verifica empiricamente il comportamento), questo modulo
implementa la strategia opposta ("requirement-driven"): a partire da un
requisito di margine minimo (tempo minimo richiesto tra il momento in cui
si sceglierebbe di attivare e il picco di accelerazione) e da una
percentuale di copertura desiderata sulla popolazione sismica di
interesse, calcola quale sia la soglia di attivazione piu' alta che
garantisce quel margine per almeno quella percentuale di sismi.

Per ogni storia temporale (dopo le stesse scremature di main.py, piu' un
filtro geografico sulla distanza stazione-localita'):
  1. Si individua l'istante t_peak in cui si verifica l'accelerazione
     massima assoluta sull'intera traccia filtrata.
  2. Si arretra di min_lead_time_s secondi da t_peak, ottenendo l'istante
     "limite" t_check = t_peak - min_lead_time_s.
  3. Si calcola tau_i = max(|segnale|) sull'intervallo [0, t_check]: e'
     il valore di soglia PIU' ALTO che, se scelto come soglia operativa,
     farebbe comunque scattare l'attivazione entro t_check (quindi con
     almeno min_lead_time_s di margine dal picco) per QUESTA specifica
     storia. Se t_check cade prima dell'inizio della traccia (il picco e'
     troppo vicino all'inizio della registrazione), il margine richiesto
     non e' raggiungibile con NESSUNA soglia per questa storia: tau_i
     viene marcato come "non raggiungibile" (-inf).
  4. Sull'insieme di tutti i tau_i della popolazione, la soglia
     raccomandata e' il valore piu' alto che lascia tau_i >= soglia per
     almeno la percentuale di copertura richiesta.

La soglia di attivazione "storica" (Config.activation_threshold, es.
0.05 m/s^2) resta comunque usata come filtro di ingresso nella
popolazione statistica: le storie che non la superano mai vengono
considerate troppo deboli per essere rilevanti e scartate a monte,
esattamente come richiesto.

Uso:
    python -m seismic_analysis.threshold_calibration \
        --site-lat 42.85 --site-lon 13.25 --radius-km 50 \
        --min-lead-time 1.5 --coverage 0.95

Eseguire con --help per l'elenco completo dei parametri configurabili.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import replace
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .calibration_plotting import plot_calibration
from .config import Config
from .filtering import apply_bandpass
from .geo import haversine_distance_km
from .io_dyna import find_dyna_files, parse_dyna_file
from .screening import check_filter_suitability, check_physical_plausibility, is_vertical_component


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", default=None, help="Cartella con i file da analizzare")
    parser.add_argument("--output-dir", default=None, help="Cartella di output")
    parser.add_argument("--site-lat", type=float, required=True, help="Latitudine della localita' [gradi decimali]")
    parser.add_argument("--site-lon", type=float, required=True, help="Longitudine della localita' [gradi decimali]")
    parser.add_argument("--radius-km", type=float, required=True,
                         help="Raggio [km] entro cui tenere le stazioni: oltre viene scartata la storia temporale")
    parser.add_argument("--activation-threshold", type=float, default=None,
                         help="Soglia [m/s^2] usata SOLO come filtro di ingresso nella popolazione "
                              "(si scartano le storie che non la superano mai; default 0.05)")
    parser.add_argument("--min-lead-time", type=float, default=None,
                         help="Tempo minimo [s] richiesto tra il momento di attivazione e il picco (default 1.5)")
    parser.add_argument("--coverage", type=float, default=None,
                         help="Copertura richiesta (0-1) sulla popolazione di sismi considerata (default 0.95)")
    parser.add_argument("--order", type=int, default=None, help="Ordine dei filtri Butterworth")
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
    parser.add_argument("--no-baseline-correction", action="store_true",
                         help="Disattiva la rimozione del trend lineare/taper prima del filtraggio")
    parser.add_argument("--taper-fraction", type=float, default=None,
                         help="Frazione di taper (Tukey) per lato applicata ai bordi prima del filtraggio (default 0.05)")
    return parser


def config_from_args(args: argparse.Namespace) -> Config:
    config = Config()
    overrides = {
        "site_lat": args.site_lat,
        "site_lon": args.site_lon,
        "radius_km": args.radius_km,
    }
    if args.input_dir is not None:
        overrides["input_dir"] = args.input_dir
    if args.output_dir is not None:
        overrides["output_dir"] = args.output_dir
    if args.activation_threshold is not None:
        overrides["activation_threshold"] = args.activation_threshold
    if args.min_lead_time is not None:
        overrides["min_lead_time_s"] = args.min_lead_time
    if args.coverage is not None:
        overrides["target_coverage"] = args.coverage
    if args.order is not None:
        overrides["filter_order"] = args.order
    if args.min_high_cut is not None:
        overrides["min_original_high_cut_hz"] = args.min_high_cut
    if args.max_plausible_accel is not None:
        overrides["max_plausible_accel_ms2"] = args.max_plausible_accel
    if args.no_baseline_correction:
        overrides["baseline_correction"] = False
    if args.taper_fraction is not None:
        overrides["taper_fraction"] = args.taper_fraction

    if args.band1 is not None or args.band2 is not None:
        bands = list(config.bands)
        if args.band1 is not None:
            bands[0] = replace(bands[0], low_hz=args.band1[0], high_hz=args.band1[1])
        if args.band2 is not None:
            bands[1] = replace(bands[1], low_hz=args.band2[0], high_hz=args.band2[1])
        overrides["bands"] = tuple(bands)

    return replace(config, **overrides)


def compute_tau(filtered_signal: np.ndarray, dt: float, min_lead_time_s: float) -> tuple[Optional[float], float, Optional[int]]:
    """Calcola tau_i per una singola storia temporale filtrata.

    Restituisce (tau_i, t_peak_s, idx_check):
      - tau_i: soglia critica per questa storia (vedi docstring del
        modulo). E' None (rappresentato poi come -inf nelle statistiche)
        se il margine richiesto non e' raggiungibile con nessuna soglia
        (il picco e' troppo vicino all'inizio della traccia).
      - t_peak_s: istante del picco assoluto [s].
      - idx_check: indice campione corrispondente a t_check (None se non
        raggiungibile)."""

    idx_peak = int(np.argmax(np.abs(filtered_signal)))
    t_peak_s = idx_peak * dt

    lead_samples = round(min_lead_time_s / dt)
    idx_check = idx_peak - lead_samples  # ultimo indice ammesso (incluso)

    if idx_check < 0:
        return None, t_peak_s, None

    window = filtered_signal[: idx_check + 1]
    tau_i = float(np.max(np.abs(window)))
    return tau_i, t_peak_s, idx_check


def compute_recommended_threshold(taus: np.ndarray, target_coverage: float) -> tuple[float, float]:
    """Calcola la soglia raccomandata e la copertura massima raggiungibile.

    taus: array di tau_i per la popolazione (puo' contenere -np.inf per
    le storie in cui il margine richiesto non e' raggiungibile con
    nessuna soglia).
    target_coverage: frazione (0-1) di popolazione da coprire.

    Il calcolo usa il metodo "nearest-rank" (non l'interpolazione lineare
    di numpy.percentile, che con valori -inf produce NaN): e' la soglia
    piu' alta per cui almeno ceil(target_coverage * n) storie hanno
    tau_i >= soglia.

    Restituisce (soglia_raccomandata, copertura_massima_raggiungibile).
    Se la soglia raccomandata risulta -inf, la copertura richiesta non e'
    raggiungibile con nessuna soglia positiva: usare
    copertura_massima_raggiungibile per sapere qual e' il tetto reale."""

    n = len(taus)
    if n == 0:
        return float("nan"), 0.0

    sorted_taus = np.sort(taus)  # ascendente; -inf finiscono in testa
    n_finite = int(np.isfinite(sorted_taus).sum())
    max_coverage = n_finite / n

    m = math.ceil(target_coverage * n)  # numero minimo di storie da coprire
    m = max(1, min(m, n))
    index = n - m  # 0-indexed
    threshold = float(sorted_taus[index])

    return threshold, max_coverage


def run(config: Config) -> dict:
    """Esegue la calibrazione per entrambe le bande. Restituisce un dict
    {band_name: {"records_df": ..., "recommended_threshold_ms2": ...,
    "max_achievable_coverage": ...}} e scrive i file di output."""

    if config.site_lat is None or config.site_lon is None or config.radius_km is None:
        raise ValueError("site_lat, site_lon e radius_km sono obbligatori per questa analisi.")

    input_dir = Path(config.input_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = find_dyna_files(input_dir)
    if not files:
        raise FileNotFoundError(f"Nessun file trovato in {input_dir}")

    max_band_high_hz = max(band.high_hz for band in config.bands)

    per_file_rows = []  # audit trail completo (una riga per file x banda)
    exclusion_rows = []  # esclusioni "a monte", indipendenti dalla banda

    candidates = []  # TimeHistory che superano le scremature indipendenti dalla banda

    for path in files:
        th = parse_dyna_file(path)

        if is_vertical_component(th, config.excluded_component):
            exclusion_rows.append({"file": path.name, "motivo": f"componente {config.excluded_component} (verticale) esclusa a priori"})
            continue

        ok, reason = check_physical_plausibility(th, config)
        if not ok:
            exclusion_rows.append({"file": path.name, "motivo": reason})
            continue

        if th.station_lat is None or th.station_lon is None:
            exclusion_rows.append({"file": path.name, "motivo": "coordinate stazione mancanti nell'header"})
            continue

        distance_km = haversine_distance_km(config.site_lat, config.site_lon, th.station_lat, th.station_lon)
        if distance_km > config.radius_km:
            exclusion_rows.append({"file": path.name, "motivo": f"stazione a {distance_km:.1f} km, oltre il raggio di {config.radius_km} km"})
            continue

        ok, reason = check_filter_suitability(th, config, max_band_high_hz)
        if not ok:
            exclusion_rows.append({"file": path.name, "motivo": reason})
            continue

        candidates.append((th, distance_km))

    # --- per banda: filtraggio, verifica soglia storica, calcolo tau_i ---
    band_results: dict[str, dict] = {}

    for band in config.bands:
        taus = []
        for th, distance_km in candidates:
            filtered = apply_bandpass(
                th.data, band.low_hz, band.high_hz, th.fs, order=config.filter_order,
                baseline_correction=config.baseline_correction,
                taper_fraction=config.taper_fraction,
            )

            ever_exceeds = bool(np.any(np.abs(filtered) >= config.activation_threshold))
            if not ever_exceeds:
                per_file_rows.append({
                    "file": th.path.name, "band": band.name, "event_id": th.event_id,
                    "magnitude": th.magnitude, "distance_km": distance_km,
                    "included_in_population": False,
                    "esclusione": f"non supera mai la soglia storica di {config.activation_threshold} m/s^2",
                    "t_peak_s": None, "feasible": None, "tau_i_ms2": None,
                })
                continue

            tau_i, t_peak_s, idx_check = compute_tau(filtered, th.dt, config.min_lead_time_s)
            feasible = tau_i is not None
            taus.append(tau_i if feasible else -np.inf)

            per_file_rows.append({
                "file": th.path.name, "band": band.name, "event_id": th.event_id,
                "magnitude": th.magnitude, "distance_km": distance_km,
                "included_in_population": True, "esclusione": None,
                "t_peak_s": t_peak_s, "feasible": feasible,
                "tau_i_ms2": tau_i if feasible else None,
            })

        taus_arr = np.array(taus, dtype=float)
        recommended_threshold, max_coverage = compute_recommended_threshold(taus_arr, config.target_coverage)

        band_results[band.name] = {
            "band": band,
            "taus": taus_arr,
            "n_population": len(taus_arr),
            "n_infeasible": int(np.sum(~np.isfinite(taus_arr))),
            "recommended_threshold_ms2": recommended_threshold,
            "max_achievable_coverage": max_coverage,
        }

    # --- output su disco ---
    per_file_df = pd.DataFrame(per_file_rows)
    exclusion_df = pd.DataFrame(exclusion_rows)

    summary_rows = []
    for band_name, res in band_results.items():
        summary_rows.append({
            "band": band_name,
            "n_popolazione": res["n_population"],
            "n_margine_non_raggiungibile": res["n_infeasible"],
            "copertura_richiesta_pct": config.target_coverage * 100,
            "copertura_massima_raggiungibile_pct": res["max_achievable_coverage"] * 100,
            "soglia_raccomandata_ms2": (
                res["recommended_threshold_ms2"] if np.isfinite(res["recommended_threshold_ms2"]) else None
            ),
            "obiettivo_raggiungibile": bool(np.isfinite(res["recommended_threshold_ms2"])),
        })
    summary_df = pd.DataFrame(summary_rows)

    per_file_df.to_csv(output_dir / "calibration_records.csv", index=False)
    exclusion_df.to_csv(output_dir / "calibration_exclusion_log.csv", index=False)
    summary_df.to_csv(output_dir / "calibration_summary.csv", index=False)

    plot_calibration(band_results, config, output_dir / "calibration_coverage_vs_threshold.png")

    return band_results, summary_df


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = config_from_args(args)

    band_results, summary_df = run(config)

    print(f"Localita': ({config.site_lat}, {config.site_lon}), raggio {config.radius_km} km")
    print(f"Margine minimo richiesto: {config.min_lead_time_s} s — copertura richiesta: {config.target_coverage*100:.0f}%")
    print(f"Soglia storica di ingresso popolazione: {config.activation_threshold} m/s^2")
    print()
    for _, row in summary_df.iterrows():
        print(f"--- Banda {row['band']} ---")
        if row["n_popolazione"] == 0:
            print("  Nessuna storia disponibile nella popolazione (raggio troppo piccolo, filtri troppo stringenti, o nessun dato nella cartella di input)")
            print()
            continue
        print(f"  Popolazione: {row['n_popolazione']} storie (di cui {row['n_margine_non_raggiungibile']} con margine non raggiungibile)")
        print(f"  Copertura massima raggiungibile: {row['copertura_massima_raggiungibile_pct']:.1f}%")
        if row["obiettivo_raggiungibile"]:
            print(f"  >>> SOGLIA RACCOMANDATA: {row['soglia_raccomandata_ms2']:.4f} m/s^2 (copre >= {config.target_coverage*100:.0f}% della popolazione)")
        else:
            print(f"  >>> OBIETTIVO NON RAGGIUNGIBILE: nessuna soglia positiva copre il {config.target_coverage*100:.0f}% richiesto "
                  f"(massimo ottenibile: {row['copertura_massima_raggiungibile_pct']:.1f}%)")
        print()

    print(f"Dettaglio per storia: {Path(config.output_dir) / 'calibration_records.csv'}")
    print(f"Log esclusioni: {Path(config.output_dir) / 'calibration_exclusion_log.csv'}")
    print(f"Riepilogo: {Path(config.output_dir) / 'calibration_summary.csv'}")
    print(f"Grafico: {Path(config.output_dir) / 'calibration_coverage_vs_threshold.png'}")


if __name__ == "__main__":
    main()
