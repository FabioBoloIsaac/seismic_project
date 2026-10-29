"""
Analisi della singola storia filtrata:
  - individuazione del primo istante in cui |segnale| supera la soglia
    di attivazione (se avviene);
  - individuazione dell'istante e del valore di accelerazione massima
    (sempre in valore assoluto) — calcolati SEMPRE, indipendentemente dal
    superamento soglia, cosi' anche le storie che non attivano restano
    confrontabili (es. per l'analisi delle discordanze tra bande, vedi
    band_comparison.py);
  - calcolo del tempo intercorso tra i due eventi, quando applicabile.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ActivationResult:
    activated: bool             # True se la soglia e' stata superata
    idx_cross: Optional[int]    # None se non attivata
    t_cross_s: Optional[float]  # None se non attivata
    idx_peak: int                # sempre disponibile
    t_peak_s: float               # sempre disponibile
    peak_value_signed: float   # valore (con segno) nell'istante di picco assoluto
    peak_value_abs: float      # |accelerazione| massima raggiunta — sempre disponibile
    delta_t_s: Optional[float]  # t_peak - t_cross, None se non attivata


def find_threshold_crossing_index(signal: np.ndarray, threshold: float) -> Optional[int]:
    """Restituisce l'indice del primo campione con |segnale| >= soglia,
    oppure None se la soglia non viene mai superata."""

    abs_signal = np.abs(signal)
    exceed = np.flatnonzero(abs_signal >= threshold)
    if exceed.size == 0:
        return None
    return int(exceed[0])


def find_absolute_peak_index(signal: np.ndarray) -> int:
    """Indice del campione con |segnale| massimo."""
    return int(np.argmax(np.abs(signal)))


def analyze_filtered_signal(signal: np.ndarray, dt: float,
                             threshold: float) -> ActivationResult:
    """Esegue l'analisi completa su una singola storia filtrata.

    A differenza delle versioni precedenti, NON restituisce piu' None se
    la soglia non viene superata: il picco (idx_peak/t_peak_s/peak_value_*)
    viene calcolato comunque, e il campo `activated` indica se la soglia
    e' stata raggiunta. idx_cross/t_cross_s/delta_t_s sono None quando
    activated=False, dato che il concetto di "tempo dal superamento
    soglia" non si applica in quel caso."""

    idx_cross = find_threshold_crossing_index(signal, threshold)
    idx_peak = find_absolute_peak_index(signal)
    activated = idx_cross is not None

    return ActivationResult(
        activated=activated,
        idx_cross=idx_cross,
        t_cross_s=(idx_cross * dt) if activated else None,
        idx_peak=idx_peak,
        t_peak_s=idx_peak * dt,
        peak_value_signed=float(signal[idx_peak]),
        peak_value_abs=float(abs(signal[idx_peak])),
        delta_t_s=((idx_peak - idx_cross) * dt) if activated else None,
    )
