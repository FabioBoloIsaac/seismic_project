"""
Lettura dei file di storia temporale in formato DYNA (ESM - Engineering
Strong Motion database).

Il file e' composto da:
  - un header testuale con righe "CHIAVE: valore"
  - a seguire, un valore per riga con la storia temporale campionata
    a passo costante SAMPLING_INTERVAL_S.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# Componenti attese nel nome file / campo STREAM dell'header.
# HNE = est-ovest, HNN = nord-sud, HNZ = verticale.
_STREAM_RE = re.compile(r"_(HN[A-Z])_")

# Fattori di conversione verso m/s^2, in base al campo header UNITS.
# Tutta la pipeline lavora internamente in m/s^2.
_UNIT_TO_MS2 = {
    "cm/s^2": 0.01,
    "cm/s2": 0.01,
    "m/s^2": 1.0,
    "m/s2": 1.0,
    "g": 9.80665,
}


@dataclass
class TimeHistory:
    """Rappresenta una storia temporale letta da un file DYNA.

    Il campo `data` e' sempre espresso in m/s^2: la conversione dall'unita'
    originale (dichiarata nell'header UNITS, tipicamente cm/s^2) avviene
    in fase di parsing."""

    path: Path
    header: dict
    data: np.ndarray          # storia temporale (accelerazione, m/s^2)
    dt: float                 # passo di campionamento [s]
    stream: str                # componente dichiarata nell'header (es. HNE)
    component_from_filename: Optional[str]  # componente dedotta dal nome file
    high_cut_hz: Optional[float]  # HIGH_CUT_FREQUENCY_HZ originale del file
    event_id: Optional[str]
    original_units: str        # unita' cosi' come dichiarata nell'header
    unit_conversion_factor: float  # fattore applicato per ottenere m/s^2
    station_lat: Optional[float]   # latitudine della stazione [gradi decimali]
    station_lon: Optional[float]   # longitudine della stazione [gradi decimali]
    event_lat: Optional[float]     # latitudine dell'epicentro [gradi decimali]
    event_lon: Optional[float]     # longitudine dell'epicentro [gradi decimali]
    magnitude: Optional[float]     # magnitudo (MAGNITUDE_W se disponibile, altrimenti MAGNITUDE_L)

    @property
    def n_samples(self) -> int:
        return len(self.data)

    @property
    def fs(self) -> float:
        """Frequenza di campionamento [Hz]."""
        return 1.0 / self.dt

    @property
    def time(self) -> np.ndarray:
        return np.arange(self.n_samples) * self.dt

    @property
    def label(self) -> str:
        """Etichetta leggibile per grafici/tabelle."""
        return f"{self.path.stem}"


def _parse_component_from_filename(path: Path) -> Optional[str]:
    match = _STREAM_RE.search(path.name)
    if match:
        return match.group(1)
    return None


def _to_float(value: str) -> Optional[float]:
    value = value.strip()
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_dyna_file(path: Path) -> TimeHistory:
    """Legge un singolo file DYNA e restituisce un oggetto TimeHistory."""

    header: dict[str, str] = {}
    data_values: list[float] = []
    reading_data = False

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n").rstrip("\r")

            if not reading_data:
                # Le righe di header hanno la forma "CHIAVE: valore".
                # La prima riga priva del separatore ':' segna l'inizio
                # della storia temporale.
                if ":" in line:
                    key, _, val = line.partition(":")
                    header[key.strip()] = val.strip()
                    continue
                else:
                    reading_data = True
                    # non c'e' un "continue" qui: la riga corrente puo'
                    # gia' contenere il primo valore dati

            stripped = line.strip()
            if stripped == "":
                continue
            try:
                data_values.append(float(stripped))
            except ValueError:
                # righe non numeriche isolate vengono ignorate
                continue

    dt_str = header.get("SAMPLING_INTERVAL_S")
    if dt_str is None:
        raise ValueError(f"Campo SAMPLING_INTERVAL_S mancante in {path}")
    dt = float(dt_str)

    stream = header.get("STREAM", "").strip()
    high_cut = _to_float(header.get("HIGH_CUT_FREQUENCY_HZ", ""))
    event_id = header.get("EVENT_ID", "").strip() or None

    original_units = header.get("UNITS", "").strip()
    units_key = original_units.lower().replace(" ", "")
    if units_key not in _UNIT_TO_MS2:
        raise ValueError(
            f"Unita' '{original_units}' non riconosciuta in {path}. "
            f"Unita' supportate: {list(_UNIT_TO_MS2)}"
        )
    conversion_factor = _UNIT_TO_MS2[units_key]

    data = np.asarray(data_values, dtype=float) * conversion_factor

    station_lat = _to_float(header.get("STATION_LATITUDE_DEGREE", ""))
    station_lon = _to_float(header.get("STATION_LONGITUDE_DEGREE", ""))
    event_lat = _to_float(header.get("EVENT_LATITUDE_DEGREE", ""))
    event_lon = _to_float(header.get("EVENT_LONGITUDE_DEGREE", ""))
    magnitude = _to_float(header.get("MAGNITUDE_W", ""))
    if magnitude is None:
        magnitude = _to_float(header.get("MAGNITUDE_L", ""))

    return TimeHistory(
        path=path,
        header=header,
        data=data,
        dt=dt,
        stream=stream,
        component_from_filename=_parse_component_from_filename(path),
        high_cut_hz=high_cut,
        event_id=event_id,
        original_units=original_units,
        unit_conversion_factor=conversion_factor,
        station_lat=station_lat,
        station_lon=station_lon,
        event_lat=event_lat,
        event_lon=event_lon,
        magnitude=magnitude,
    )


def find_dyna_files(input_dir: Path, patterns: tuple[str, ...] = ("*.txt", "*.ASC", "*.asc")) -> list[Path]:
    """Elenca (in ordine alfabetico) i file da analizzare in una cartella.

    I file DYNA scaricati manualmente hanno spesso estensione .txt, mentre
    quelli scaricati automaticamente dal web service ESM hanno estensione
    .ASC: per questo vengono cercati entrambi i pattern di default."""

    found: set[Path] = set()
    for pattern in patterns:
        found.update(Path(input_dir).glob(pattern))
    return sorted(found)
