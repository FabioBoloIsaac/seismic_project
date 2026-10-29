"""
Download automatico di eventi sismici italiani e delle relative forme
d'onda accelerometriche, da integrare nella cartella dati condivisa sul
NAS (vedi Config.input_dir in config.py).

Fonti:
  - ISIDe (INGV) - catalogo parametrico completo dei terremoti italiani
    https://terremoti.ingv.it/iside
  - ESM (Engineering Strong-Motion database) - forme d'onda in formato
    DYNA ASCII, la stessa fonte dei file gia' presenti in questo progetto
    https://esm-db.eu

Uso tipico:

    # 1. Elenca gli eventi italiani con M >= 4.5 in un certo periodo
    python -m seismic_analysis.download_data list-events \
        --start 2016-08-01 --end 2016-12-31 --min-magnitude 4.5

    # 2. Scarica le forme d'onda ESM GREZZE (non filtrate, processing-type
    #    CV) per quegli eventi e le salva nella cartella dati sul NAS
    #    (default, vedi Config.input_dir), pronte per essere filtrate da
    #    `python -m seismic_analysis.main`
    python -m seismic_analysis.download_data download-waveforms \
        --start 2016-08-01 --end 2016-12-31 --min-magnitude 4.5

Note:
  - I dati "pubblici" (non embargati) di ESM sono scaricabili con una
    semplice richiesta HTTP GET, senza registrazione.
  - Per default vengono scaricati i dati grezzi non filtrati
    (--processing-type CV): nessun filtro applicato in origine, nessun
    ritaglio, cosi' il progetto puo' applicare i propri due Butterworth
    su tutta la banda disponibile. Con dati CV la "seconda scrematura"
    della pipeline (seismic_analysis/main.py) non scarta piu' in base a
    un filtro originale (che non esiste), ma verifica che la frequenza
    di campionamento del file sia sufficiente per le bande richieste.
  - ATTENZIONE: la frequenza di campionamento NON e' sempre 200 Hz. Varia
    a seconda della stazione/rete (osservati anche 100 Hz e 125 Hz): il
    parser legge sempre il valore corretto dall'header SAMPLING_INTERVAL_S
    di ciascun file, quindi non va assunta a priori.
  - Il servizio ESM limita ogni richiesta eventdata a 1000 forme d'onda:
    per eventi molto grandi lo script filtra comunque per network/stazione
    se specificato, e scarica un evento alla volta.
  - E' buona norma non bombardare il servizio con troppe richieste
    ravvicinate: lo script inserisce una piccola pausa tra un evento e
    l'altro (--sleep, default 1s).
  - Se un evento risulta gia' (almeno parzialmente) scaricato in
    precedenza nella cartella di destinazione, la richiesta HTTP per
    quell'evento viene saltata del tutto: comodo per riprendere un
    download interrotto senza rifare tutto da capo. Per forzare comunque
    un nuovo download/sovrascrittura, usare --force.
"""

from __future__ import annotations

import argparse
import io
import time
import zipfile
from pathlib import Path

import requests

from .config import Config

ISIDE_EVENT_URL = "http://webservices.ingv.it/fdsnws/event/1/query"
ESM_EVENT_URL = "https://esm-db.eu/fdsnws/event/1/query"
ESM_EVENTDATA_URL = "https://esm-db.eu/esmws/eventdata/1/query"

# Bounding box che copre l'intero territorio italiano (con margine).
ITALY_BBOX = dict(minlatitude=35.0, maxlatitude=47.5, minlongitude=6.0, maxlongitude=19.0)


def list_events(start: str, end: str, min_magnitude: float = 4.0,
                 bbox: dict | None = None) -> list[dict]:
    """Interroga ISIDe (fdsnws-event, catalogo INGV) e restituisce la
    lista di eventi italiani nel periodo/magnitudo richiesti.

    Utile per farsi un'idea di *quanti* terremoti sono avvenuti (il
    catalogo ISIDe è molto più esteso di ESM: include anche eventi di
    magnitudo molto bassa). Gli ID restituiti sono identificativi interni
    INGV e NON sono detti coincidere con l'ID richiesto dal servizio
    forme d'onda ESM: per scaricare le forme d'onda usare
    `list_esm_events` / `download-waveforms`.
    """

    params = {
        "starttime": start,
        "endtime": end,
        "minmagnitude": min_magnitude,
        "format": "text",
        "orderby": "time",
        **(bbox or ITALY_BBOX),
    }
    resp = requests.get(ISIDE_EVENT_URL, params=params, timeout=60)
    resp.raise_for_status()

    lines = resp.text.strip().splitlines()
    if not lines or not lines[0].startswith("#"):
        return []

    header = [h.strip() for h in lines[0].lstrip("#").split("|")]
    events = []
    for line in lines[1:]:
        values = line.split("|")
        events.append(dict(zip(header, values)))
    return events


def list_esm_events(start: str, end: str, min_magnitude: float = 4.0,
                     bbox: dict | None = None) -> list[dict]:
    """Interroga il catalogo eventi *di ESM stesso* (fdsnws-event su
    esm-db.eu). Gli ID restituiti sono garantiti compatibili con il
    servizio forme d'onda ESM (`download_esm_waveforms`), quindi e'
    questa la funzione da usare per il download automatico delle tracce.

    Nota: ESM copre solo eventi con magnitudo >= 4.0 (limite del database),
    quindi e' inutile impostare min_magnitude sotto 4.0."""

    params = {
        "starttime": start,
        "endtime": end,
        "minmagnitude": max(min_magnitude, 4.0),
        "format": "text",
        "orderby": "time",
        **(bbox or ITALY_BBOX),
    }
    resp = requests.get(ESM_EVENT_URL, params=params, timeout=60)
    resp.raise_for_status()

    lines = resp.text.strip().splitlines()
    if not lines:
        return []

    # A differenza di ISIDe, l'header ESM non e' prefissato da '#' e usa
    # nomi di colonna propri (ev_id, mag_value, event_name, ...).
    header = [h.strip().lstrip("#") for h in lines[0].split("|")]
    events = []
    for line in lines[1:]:
        if not line.strip():
            continue
        values = line.split("|")
        events.append(dict(zip(header, values)))
    return events


def existing_tracks_for_event(output_dir: Path, event_id: str) -> list[Path]:
    """Restituisce i file gia' presenti in output_dir riconducibili
    all'evento event_id (qualunque sia stazione/canale), cercando l'ID
    evento nel nome file (presente in tutti i file DYNA scaricati da ESM,
    es. '...D.EMSC-20161026_0000077.ACC.CV.ASC')."""

    if not output_dir.exists():
        return []
    return list(output_dir.glob(f"*{event_id}*"))


def download_esm_waveforms(event_id: str, output_dir: Path,
                            network: str | None = None,
                            station: str | None = None,
                            data_type: str = "ACC",
                            processing_type: str = "CV",
                            force: bool = False) -> tuple[list[Path], list[Path]]:
    """Scarica dal database ESM le forme d'onda (formato ASCII/DYNA) di un
    singolo evento e le estrae in output_dir.

    Restituisce una tupla (nuovi_file, file_gia_presenti_saltati). Se
    force=False (default), i singoli file gia' presenti in output_dir non
    vengono ri-scaricati/sovrascritti (utile per riprendere un download
    interrotto): la richiesta HTTP viene comunque fatta (per elencare le
    tracce dell'evento), ma il salvataggio su disco viene saltato per i
    file gia' presenti. Per saltare l'evento *prima* ancora di fare la
    richiesta HTTP, vedere `existing_tracks_for_event` usato in main().

    processing_type:
      - 'CV' (default): dato grezzo ("converted"), solo convertito da
        conteggi digitali a cm/s^2. Nessun filtro applicato, nessun
        ritaglio: e' il dato da usare se si vuole filtrare per conto
        proprio (campi FILTER_*/HIGH_CUT_FREQUENCY_HZ dell'header vuoti).
      - 'MP': dato processato manualmente da un analista (correzione di
        linea di base, filtraggio, ritaglio attorno all'evento). E' il
        formato dei file originariamente inclusi in questo progetto.
      - 'AP': dato processato automaticamente (via ESMpro), via di mezzo
        tra CV e MP in termini di qualita'/revisione."""

    params = {
        "eventid": event_id,
        "format": "ascii",
        "data-type": data_type,
        "processing-type": processing_type,
    }
    if network:
        params["network"] = network
    if station:
        params["station"] = station

    resp = requests.get(ESM_EVENTDATA_URL, params=params, timeout=120)

    # ESM restituisce 204/404 (o un corpo non-zip) quando l'evento non ha
    # forme d'onda disponibili: in tal caso si salta senza errore.
    if resp.status_code != 200 or not resp.content.startswith(b"PK"):
        return [], []

    output_dir.mkdir(parents=True, exist_ok=True)
    new_files: list[Path] = []
    skipped_files: list[Path] = []
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        for member in zf.namelist():
            target = output_dir / Path(member).name
            if target.exists() and not force:
                skipped_files.append(target)
                continue
            with zf.open(member) as src, open(target, "wb") as dst:
                dst.write(src.read())
            new_files.append(target)
    return new_files, skipped_files


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list-events", help="Elenca gli eventi italiani da ISIDe (senza scaricare forme d'onda)")
    p_list.add_argument("--start", required=True, help="Data inizio, es. 2016-08-01")
    p_list.add_argument("--end", required=True, help="Data fine, es. 2016-12-31")
    p_list.add_argument("--min-magnitude", type=float, default=4.0)

    p_dl = sub.add_parser("download-waveforms", help="Scarica le forme d'onda ESM per gli eventi trovati")
    p_dl.add_argument("--start", required=True)
    p_dl.add_argument("--end", required=True)
    p_dl.add_argument("--min-magnitude", type=float, default=4.0,
                       help="ESM copre eventi con M >= 4.0: valori inferiori non produrranno risultati")
    p_dl.add_argument("--output-dir", default=Config().input_dir,
                       help="Cartella di destinazione (default: la cartella dati configurata in config.py, sul NAS)")
    p_dl.add_argument("--network", default=None, help="Filtra per rete sismica (es. IT, 3A). Default: tutte")
    p_dl.add_argument("--station", default=None, help="Filtra per stazione (es. MZ01). Default: tutte")
    p_dl.add_argument("--processing-type", default="CV", choices=["CV", "MP", "AP"],
                       help="'CV'=grezzo non filtrato (default, consigliato per filtrare per conto proprio), "
                            "'MP'=processato manualmente, 'AP'=processato automaticamente")
    p_dl.add_argument("--sleep", type=float, default=1.0, help="Pausa [s] tra una richiesta e l'altra")
    p_dl.add_argument("--force", action="store_true",
                       help="Riscarica/sovrascrive anche gli eventi/file gia' presenti (default: vengono saltati)")

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.command == "list-events":
        events = list_events(args.start, args.end, args.min_magnitude)
        print(f"Trovati {len(events)} eventi.")
        for ev in events:
            print(f"  {ev['EventID']}  {ev['Time']}  M{ev['Magnitude']} ({ev['MagType']})  {ev['EventLocationName']}")

    elif args.command == "download-waveforms":
        events = list_esm_events(args.start, args.end, args.min_magnitude)
        print(f"Trovati {len(events)} eventi nel catalogo ESM. Scarico le forme d'onda...")

        output_dir = Path(args.output_dir)
        total_new_files = 0
        total_skipped_files = 0
        n_events_with_data = 0
        n_events_skipped = 0

        for ev in events:
            event_id = ev["ev_id"]

            # Se l'evento risulta gia' (almeno parzialmente) presente in
            # output_dir e non e' stato chiesto --force, si salta del
            # tutto la richiesta HTTP per quell'evento.
            if not args.force:
                existing = existing_tracks_for_event(output_dir, event_id)
                if existing:
                    n_events_skipped += 1
                    print(f"  [==] {event_id}: gia' presente ({len(existing)} tracce trovate), salto")
                    continue

            new_files, skipped_files = download_esm_waveforms(
                event_id, output_dir,
                network=args.network, station=args.station,
                processing_type=args.processing_type,
                force=args.force,
            )
            if new_files or skipped_files:
                n_events_with_data += 1
                total_new_files += len(new_files)
                total_skipped_files += len(skipped_files)
                msg = f"  [OK] {event_id} (M{ev['mag_value']}, {ev['event_name']}): {len(new_files)} tracce scaricate"
                if skipped_files:
                    msg += f" ({len(skipped_files)} gia' presenti, saltate)"
                print(msg)
            else:
                print(f"  [--] {event_id}: nessuna forma d'onda disponibile con i filtri richiesti")
            time.sleep(args.sleep)

        print(
            f"\nCompletato: {total_new_files} nuovi file scaricati "
            f"({total_skipped_files} gia' presenti saltati a livello di singolo file, "
            f"{n_events_skipped} eventi saltati interamente perche' gia' scaricati) "
            f"da {n_events_with_data}/{len(events)} eventi con dati disponibili."
        )
        print(f"Salvati in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
