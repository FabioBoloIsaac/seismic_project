"""
Generazione dello scatter plot riassuntivo:
  X = tempo tra superamento soglia e accelerazione massima [s]
  Y = accelerazione massima raggiunta (valore assoluto) [m/s^2]
Colore diverso per le due bande di filtraggio (20 Hz / 40 Hz).

Riceve il DataFrame risultati COMPLETO (idealmente con la colonna
`activated`, presente nei results.csv prodotti dalle versioni recenti di
main.py): internamente si tengono solo le storie attivate per il disegno
(il delta_t ha senso solo in quel caso), ma l'informazione sulle altre
bande viene usata per individuare le storie "orfane" — attivate in questa
banda ma NON nell'altra — che vengono disegnate in NERO invece che nel
colore della banda, cosi' si distinguono a colpo d'occhio le attivazioni
concordi da quelle discordanti. Se la colonna `activated` non e' presente
(CSV di versioni precedenti, dove solo le storie attivate erano salvate),
l'individuazione delle orfane si basa sulla semplice presenza/assenza del
file nelle righe dell'altra banda, che per quel formato equivale alla
stessa cosa.

Se Config.connect_band_points e' True (default False, CLI --connect-bands),
per ogni storia attivata in TUTTE le bande configurate (non per le orfane,
che non hanno nulla da collegare) viene disegnata una linea sottile e
semitrasparente che unisce il pallino di una banda con quello dell'altra:
utile per vedere a colpo d'occhio quanto si sposta lo stesso evento tra
una banda e l'altra. Disattivato di default perche' su popolazioni grandi
puo' affollare il grafico.

Per ciascuna banda possono essere disegnate fino a due linee verticali
tratteggiate indipendenti (vedi Config.delta_t_percentile e
Config.delta_t_fixed_threshold_s, in config.py):
  1. Linea a PERCENTILE fisso: si sceglie una percentuale (es. 5) e si
     calcola il valore di delta_t sotto il quale cade quella percentuale
     di storie (con 5, il 95% delle storie ha delta_t maggiore del
     valore trovato).
  2. Linea a TEMPO fisso: al contrario, si fissa il tempo (in secondi) e
     si calcola quale percentuale di storie ha delta_t maggiore di quel
     tempo.
Ciascuna delle due e' indipendente e puo' essere disattivata impostando
il relativo parametro di Config a -1 (valore fittizio, dato che ne' un
percentile ne' un tempo possono essere negativi). Le linee (e i conteggi
"n=" in legenda) tengono conto di TUTTE le storie della banda, incluse
le orfane: il colore nero e' solo visivo, non esclude nulla dalle
statistiche.

Le due linee vengono calcolate SOLO sulle storie con accelerazione di
picco >= Config.line_min_peak_accel_ms2 (seconda soglia, indipendente da
activation_threshold): serve a escludere dalle statistiche delle linee le
storie piu' deboli/vicine al rumore, che altrimenti possono dominare il
calcolo con delta_t quasi nulli poco rappresentativi. Lo SCATTER invece
mostra sempre tutti i punti della banda, filtro o non filtro: la seconda
soglia influenza solo le due linee, non quali punti sono disegnati.

Gli assi sono limitati per leggibilita': l'asse X parte sempre da 0 (i
valori negativi di delta_t, se presenti, escono dal grafico ma restano nei
CSV di output) ed e' tagliato a destra al percentile scelto in
Config.x_axis_clip_percentile (default 95° percentile di delta_t, su
tutte le bande insieme — alzarlo "allarga" il grafico verso destra);
l'asse Y e' tagliato a un valore "ragionevole" con la regola di Tukey,
cosi' pochi outlier molto piu' alti della maggior parte dei punti non
schiacciano la scala. Eventuali punti fuori da questi limiti non vengono
mostrati, ma restano comunque nei CSV. Le percentuali riportate in
legenda (sia per le linee verticali sia altrove) sono sempre calcolate sui
dati completi (o sul sottoinsieme filtrato per le linee, vedi sopra), non
sulla sola porzione visibile del grafico.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.collections import LineCollection

from .config import Config

X_AXIS_MARGIN_FACTOR = 1.1

# Percentile dell'accelerazione di picco (su tutte le bande insieme) usato
# per fissare il limite superiore dell'asse Y: si usa la regola di Tukey
# (basata su quartili/IQR, lo standard per individuare outlier nei boxplot)
# perche' e' robusta anche quando gli outlier sono una frazione non
# piccolissima dei punti (un semplice percentile alto, es. il 95°, puo'
# cadere esso stesso dentro il gruppo di outlier se sono piu' del 5%).
Y_AXIS_IQR_WHISKER_FACTOR = 1.5

# Valore fittizio che, assegnato a Config.delta_t_percentile o a
# Config.delta_t_fixed_threshold_s, disattiva la relativa linea verticale.
DISABLED_SENTINEL = -1

# Colore usato per le attivazioni "orfane" (senza corrispettivo nelle
# altre bande configurate), al posto del colore della propria banda.
ORPHAN_COLOR = "black"


def _active_files_by_band(results_df: pd.DataFrame) -> dict[str, set]:
    """File attivati per ciascuna banda. Se la colonna `activated` e'
    presente si filtra su quella; altrimenti (CSV di versioni precedenti,
    dove solo le storie attivate venivano salvate) si usa la semplice
    presenza della riga, che per quel formato equivale alla stessa cosa."""

    source = results_df[results_df["activated"]] if "activated" in results_df.columns else results_df
    return {band: set(source.loc[source["band"] == band, "file"]) for band in source["band"].unique()}


def plot_results(results_df: pd.DataFrame, config: Config, output_path: Path,
                  summary_df: Optional[pd.DataFrame] = None) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))

    active_files_by_band = _active_files_by_band(results_df)
    all_band_names = [b.name for b in config.bands]

    # Sottoinsieme "da disegnare": solo storie attivate (il delta_t ha
    # senso solo in quel caso). Con la colonna `activated` assente
    # (CSV di versioni precedenti) si assume che tutte le righe lo siano
    # gia' (comportamento storico).
    shown_df = results_df[results_df["activated"]] if "activated" in results_df.columns else results_df

    # --- linea sottile che collega il pallino di una banda con quello
    # dell'altra per la STESSA storia (solo storie attivate in TUTTE le
    # bande, non per le orfane: non c'e' nulla da collegare per quelle).
    # Disegnata per prima cosi' resta sotto i pallini (zorder piu' basso).
    if config.connect_band_points and len(all_band_names) >= 2 and not shown_df.empty:
        pivot_x = shown_df.pivot_table(index="file", columns="band", values="delta_t_s", aggfunc="first")
        pivot_y = shown_df.pivot_table(index="file", columns="band", values="peak_abs_accel_ms2", aggfunc="first")
        if all(b in pivot_x.columns for b in all_band_names):
            complete_mask = pivot_x[all_band_names].notna().all(axis=1) & pivot_y[all_band_names].notna().all(axis=1)
            px = pivot_x.loc[complete_mask, all_band_names].to_numpy()
            py = pivot_y.loc[complete_mask, all_band_names].to_numpy()
            if len(px):
                segments = []
                for i in range(len(all_band_names) - 1):
                    segments.extend(np.stack([px[:, i:i + 2], py[:, i:i + 2]], axis=-1))
                ax.add_collection(LineCollection(segments, colors="gray", linewidths=0.6, alpha=0.35, zorder=1))
                ax.plot([], [], color="gray", linewidth=1.2, alpha=0.6,
                        label=f"Collegamento stessa storia tra bande — n={len(px)}")

    for band in config.bands:
        subset = shown_df[shown_df["band"] == band.name]
        if subset.empty:
            continue

        # Una storia e' "orfana" in questa banda se non risulta attivata
        # in TUTTE le altre bande configurate.
        other_bands = [b for b in all_band_names if b != band.name]
        is_orphan = ~subset["file"].apply(
            lambda f: all(f in active_files_by_band.get(b, set()) for b in other_bands)
        ) if other_bands else pd.Series(False, index=subset.index)

        matched = subset[~is_orphan]
        orphan = subset[is_orphan]

        if not matched.empty:
            ax.scatter(
                matched["delta_t_s"], matched["peak_abs_accel_ms2"],
                color=band.color,
                label=f"Banda {band.name} ({band.low_hz}-{band.high_hz} Hz) — n={len(subset)} (di cui {len(orphan)} senza corrispettivo)"
                if not orphan.empty else f"Banda {band.name} ({band.low_hz}-{band.high_hz} Hz) — n={len(subset)}",
                alpha=0.8, edgecolors="k", linewidths=0.4, s=60, zorder=2,
            )
        if not orphan.empty:
            ax.scatter(
                orphan["delta_t_s"], orphan["peak_abs_accel_ms2"],
                color=ORPHAN_COLOR,
                label=f"Banda {band.name}: senza corrispettivo nelle altre bande — n={len(orphan)}",
                alpha=0.8, edgecolors="k", linewidths=0.4, s=60, marker="o", zorder=2,
            )

        # Sottoinsieme usato SOLO per il calcolo delle due linee: esclude
        # le storie con accelerazione di picco sotto la seconda soglia.
        # Con line_min_peak_accel_ms2 = 0 (default) non filtra nulla.
        # Include sia le storie "matched" sia le "orfane" (il nero e'
        # solo un colore, non un'esclusione dalle statistiche).
        line_subset = subset[subset["peak_abs_accel_ms2"] >= config.line_min_peak_accel_ms2]

        if line_subset.empty:
            # Nessuna storia sopra la seconda soglia in questa banda:
            # non si puo' disegnare nessuna linea, lo si segnala in legenda.
            if config.delta_t_percentile != DISABLED_SENTINEL or config.delta_t_fixed_threshold_s != DISABLED_SENTINEL:
                ax.plot([], [], " ", label=(
                    f"Banda {band.name}: nessuna storia con picco >= "
                    f"{config.line_min_peak_accel_ms2:g} m/s² (linee non calcolabili)"
                ))
            continue

        delta_t_values = line_subset["delta_t_s"]
        suffix = f" [su {len(line_subset)}/{len(subset)} storie con picco >= {config.line_min_peak_accel_ms2:g} m/s²]" \
            if config.line_min_peak_accel_ms2 > 0 else ""

        # --- Linea 1: percentile fisso -> calcola il tempo ---
        if config.delta_t_percentile != DISABLED_SENTINEL:
            p_line = np.percentile(delta_t_values, config.delta_t_percentile)
            pct_right = 100 - config.delta_t_percentile
            ax.axvline(
                p_line,
                color=band.color,
                linestyle="--",
                linewidth=1.5,
                alpha=0.8,
                label=(
                    f"P{config.delta_t_percentile:g} banda {band.name}: {p_line:.2f} s "
                    f"({pct_right:.0f}% dei punti a destra){suffix}"
                ),
            )

        # --- Linea 2: tempo fisso -> calcola la percentuale ---
        if config.delta_t_fixed_threshold_s != DISABLED_SENTINEL:
            t_fixed = config.delta_t_fixed_threshold_s
            pct_over = 100 * (delta_t_values > t_fixed).mean()
            ax.axvline(
                t_fixed,
                color=band.color,
                linestyle=":",
                linewidth=1.5,
                alpha=0.8,
                label=(
                    f"t={t_fixed:g}s banda {band.name}: "
                    f"{pct_over:.1f}% dei punti con delta_t > soglia{suffix}"
                ),
            )

    ax.set_xlabel("Tempo tra superamento soglia e accelerazione massima [s]")
    ax.set_ylabel("Accelerazione massima assoluta [m/s$^2$]")
    ax.set_title("Superamento soglia vs. picco di accelerazione")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # --- limiti degli assi ---
    # X: parte sempre da 0; a destra e' tagliata al percentile scelto (+
    # margine) cosi' la coda di eventi con delta_t molto lungo non
    # schiaccia visivamente il resto dei punti (incluse le linee sopra).
    # Se non ci sono veri outlier, non si taglia nulla di significativo.
    all_delta_t = shown_df["delta_t_s"]
    if not all_delta_t.empty:
        x_clip = np.percentile(all_delta_t, config.x_axis_clip_percentile) * X_AXIS_MARGIN_FACTOR
        x_top = min(x_clip, all_delta_t.max()) if x_clip < all_delta_t.max() else all_delta_t.max()
        ax.set_xlim(left=0, right=x_top * 1.02)
    else:
        ax.set_xlim(left=0)

    # Y: tagliata a un valore ragionevole calcolato con la regola di Tukey
    # (Q3 + 1.5*IQR), cosi' un gruppo di outlier molto piu' alti della
    # maggior parte dei punti non schiaccia la scala. Se non ci sono veri
    # outlier (tutti i punti gia' entro quel limite), non si taglia nulla.
    all_peaks = shown_df["peak_abs_accel_ms2"]
    if not all_peaks.empty:
        q1, q3 = np.percentile(all_peaks, [25, 75])
        iqr = q3 - q1
        y_clip = q3 + Y_AXIS_IQR_WHISKER_FACTOR * iqr
        y_top = min(y_clip, all_peaks.max()) if y_clip < all_peaks.max() else all_peaks.max()
        ax.set_ylim(bottom=0, top=y_top * 1.05)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
