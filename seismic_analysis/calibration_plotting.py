"""
Grafico "copertura vs soglia" per threshold_calibration.py.

Per ciascuna banda, mostra la funzione a gradini: per ogni possibile
valore di soglia theta sull'asse X, quale percentuale della popolazione
(asse Y) avrebbe raggiunto il margine minimo richiesto scegliendo quella
soglia. E' una funzione non crescente in theta (soglie piu' alte
coprono meno storie). La soglia raccomandata e' il punto in cui questa
curva incrocia la copertura richiesta.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from .config import Config


def plot_calibration(band_results: dict, config: Config, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))

    any_finite_tau = False

    for band_name, res in band_results.items():
        band = res["band"]
        taus = res["taus"]
        n = res["n_population"]
        if n == 0:
            continue

        finite_taus = taus[np.isfinite(taus)]
        if finite_taus.size:
            any_finite_tau = True

        # Costruzione della funzione a gradini "copertura raggiunta se si
        # sceglie soglia = theta": ordinando i tau (compresi i -inf) in
        # modo decrescente, il k-esimo valore e' la soglia piu' alta che
        # garantisce che almeno k storie su n abbiano tau_i >= soglia.
        sorted_desc = np.sort(taus)[::-1]
        coverage_pct = 100 * np.arange(1, n + 1) / n

        # Per il grafico si mostra solo la parte con soglia finita e >= 0
        # (i punti a soglia -inf rappresentano "nessuna soglia positiva
        # copre oltre questo punto" e vengono lasciati fuori dall'asse,
        # ma la copertura massima raggiungibile e' comunque annotata).
        finite_mask = np.isfinite(sorted_desc)
        x = sorted_desc[finite_mask]
        y = coverage_pct[finite_mask]

        if x.size:
            ax.step(x, y, where="post", color=band.color, linewidth=2,
                    label=f"Banda {band_name} ({band.low_hz}-{band.high_hz} Hz) — n={n}")

        rec = res["recommended_threshold_ms2"]
        if np.isfinite(rec):
            ax.axvline(rec, color=band.color, linestyle="--", linewidth=1.3, alpha=0.7)
            ax.plot([rec], [config.target_coverage * 100], marker="o", color=band.color,
                    markersize=8, markeredgecolor="k", zorder=5)

    ax.axhline(config.target_coverage * 100, color="gray", linestyle=":", linewidth=1.3,
               label=f"Copertura richiesta: {config.target_coverage*100:.0f}%")

    ax.set_xlabel("Soglia di attivazione ipotizzata [m/s$^2$]")
    ax.set_ylabel("Percentuale di sismi coperti [%]")
    ax.set_title(
        f"Copertura vs soglia — margine minimo richiesto: {config.min_lead_time_s:g} s"
    )
    ax.set_ylim(0, 105)
    if any_finite_tau:
        ax.set_xlim(left=0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
