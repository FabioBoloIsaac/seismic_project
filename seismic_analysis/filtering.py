"""
Filtraggio passa-banda Butterworth applicato a fase nulla (filtfilt),
in modo da non introdurre sfasamento temporale che comprometterebbe la
stima del tempo tra superamento soglia e picco di accelerazione.

Prima del filtraggio viene applicato un preprocessing standard (rimozione
del trend lineare, che include la media, e taper ai bordi) per evitare
transitori del filtro quando il segnale grezzo non parte/finisce a media
nulla o presenta un trend residuo — tipico dei dati non processati (es.
ESM 'CV': header 'BASELINE NOT REMOVED').
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt, detrend
from scipy.signal.windows import tukey


def design_bandpass_sos(low_hz: float, high_hz: float, fs: float, order: int = 1):
    """Progetta un filtro Butterworth passa-banda in forma SOS (piu' stabile
    numericamente rispetto ai coefficienti b/a per ordini elevati)."""

    nyquist = fs / 2.0
    if high_hz >= nyquist:
        raise ValueError(
            f"La frequenza di taglio superiore ({high_hz} Hz) deve essere "
            f"inferiore alla frequenza di Nyquist ({nyquist} Hz)."
        )
    low = low_hz / nyquist
    high = high_hz / nyquist
    return butter(order, [low, high], btype="bandpass", output="sos")


def preprocess_signal(signal: np.ndarray, taper_fraction: float = 0.05) -> np.ndarray:
    """Rimuove il trend lineare (include la media) e applica un taper
    (finestra di Tukey) ai bordi del segnale.

    Serve a evitare transitori/ringing del filtro passa-banda quando il
    segnale grezzo non e' a media nulla o presenta un offset/trend
    residuo (tipico dei dati non processati, es. ESM 'CV'): un gradino o
    una pendenza non nulla ai bordi, filtrata con filtfilt, puo' generare
    oscillazioni spurie proprio all'inizio/fine della traccia filtrata.

    taper_fraction: frazione (0-0.5) della lunghezza del segnale, per lato,
    su cui applicare la rastremazione del taper di Tukey. 0 disattiva il
    taper (viene comunque rimosso il trend lineare)."""

    detrended = detrend(signal, type="linear")
    if taper_fraction > 0:
        window = tukey(len(detrended), alpha=2 * taper_fraction)
        detrended = detrended * window
    return detrended


def apply_bandpass(signal: np.ndarray, low_hz: float, high_hz: float,
                    fs: float, order: int = 1,
                    baseline_correction: bool = True,
                    taper_fraction: float = 0.05) -> np.ndarray:
    """Applica il filtro passa-banda a fase nulla al segnale.

    Se baseline_correction=True (default), prima del filtraggio il
    segnale viene "ripulito" con preprocess_signal (rimozione trend
    lineare + taper ai bordi)."""

    if baseline_correction:
        signal = preprocess_signal(signal, taper_fraction=taper_fraction)

    sos = design_bandpass_sos(low_hz, high_hz, fs, order=order)
    return sosfiltfilt(sos, signal)
