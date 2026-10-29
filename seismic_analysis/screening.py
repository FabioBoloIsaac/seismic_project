"""
Logica di scrematura condivisa tra la pipeline di analisi (main.py) e la
calibrazione della soglia (threshold_calibration.py), per garantire lo
stesso comportamento in entrambi gli strumenti.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .config import Config
from .io_dyna import TimeHistory


def is_vertical_component(th: TimeHistory, excluded_component: str) -> bool:
    """Vero se la storia temporale appartiene alla componente da escludere
    (default: verticale, 'Z'), verificando sia il nome file sia l'header."""

    from_filename = th.component_from_filename or ""
    from_header = th.stream or ""
    return from_filename.endswith(excluded_component) or from_header.endswith(excluded_component)


def check_physical_plausibility(th: TimeHistory, config: Config) -> Tuple[bool, Optional[str]]:
    """Verifica che il valore assoluto massimo del segnale GREZZO (prima
    di qualunque filtro) sia fisicamente plausibile per un'accelerazione
    sismica al suolo. Protegge da glitch del digitalizzatore, tipici dei
    dati non processati (es. ESM 'CV', privi di controllo qualita'
    automatico) — osservati in dataset reali fino a ~2 milioni di m/s^2.

    Restituisce (ok, motivo_esclusione_o_None)."""

    if th.data.size == 0:
        return False, "storia temporale vuota"

    max_abs = float(np.max(np.abs(th.data)))
    if max_abs > config.max_plausible_accel_ms2:
        return False, (
            f"valore grezzo implausibile: {max_abs:.1f} m/s^2 supera la soglia di "
            f"plausibilita' fisica ({config.max_plausible_accel_ms2:g} m/s^2) — "
            f"probabile glitch del digitalizzatore"
        )
    return True, None


def check_filter_suitability(th: TimeHistory, config: Config,
                              max_band_high_hz: float) -> Tuple[bool, Optional[str]]:
    """Verifica se il file e' idoneo al filtraggio richiesto (seconda
    scrematura). Restituisce (ok, motivo_esclusione_o_None).

    - Se il file dichiara un HIGH_CUT_FREQUENCY_HZ originale (dati gia'
      processati, es. ESM 'MP'): escluso se quel filtro era piu' stretto
      della soglia minima configurata.
    - Se il campo e' assente (dati grezzi, es. ESM 'CV'): si verifica
      invece che la frequenza di campionamento sia sufficiente per la
      banda piu' esigente richiesta (Nyquist > frequenza di taglio
      superiore piu' alta tra le bande)."""

    if th.high_cut_hz is not None:
        if th.high_cut_hz < config.min_original_high_cut_hz:
            return False, (
                f"HIGH_CUT_FREQUENCY_HZ originale = {th.high_cut_hz} "
                f"< soglia minima {config.min_original_high_cut_hz} Hz"
            )
        return True, None

    nyquist_hz = th.fs / 2.0
    if nyquist_hz <= max_band_high_hz:
        return False, (
            f"frequenza di campionamento insufficiente: fs={th.fs:.1f} Hz "
            f"(Nyquist={nyquist_hz:.1f} Hz) <= banda piu' alta richiesta "
            f"({max_band_high_hz} Hz)"
        )
    return True, None
