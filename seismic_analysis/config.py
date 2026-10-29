"""
Configurazione centrale dell'analisi.

Tutti i parametri "scientifici" dell'analisi (ordine dei filtri, bande di
frequenza, soglia di attivazione, soglia di scarto sui file gia' filtrati)
sono raccolti qui in modo da poterli modificare in un unico punto, oppure
sovrascriverli da linea di comando (vedi main.py --help).
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass(frozen=True)
class FilterBand:
    """Definisce una delle due bande passa-banda Butterworth da applicare."""

    name: str          # etichetta breve usata nei risultati e nei grafici (es. "20Hz")
    low_hz: float       # frequenza di taglio inferiore [Hz]
    high_hz: float      # frequenza di taglio superiore [Hz]
    color: str          # colore usato nello scatter plot per distinguere la banda


@dataclass
class Config:
    # --- percorsi ---
    # Cartella con i file DYNA (.txt/.ASC) da analizzare e verso cui lo
    # script di download salva le nuove forme d'onda. E' un percorso di
    # rete (NAS), condiviso e indipendente dalla cartella del progetto:
    # va aggiornato qui se il percorso del NAS dovesse cambiare.
    input_dir: str = r"Z:\01_SVILUPPO PRODOTTO\26_09_ACT_LOGIC\01_DATI\Terremoti scaricati\data"
    output_dir: str = "output"

    # --- criteri di selezione dei file (prima scrematura) ---
    # Direzione da escludere sempre (verticale). Confrontata sia sul nome
    # file (gruppo HN?) sia, come controllo incrociato, sul campo header STREAM.
    excluded_component: str = "Z"

    # --- criteri di selezione dei file (seconda scrematura) ---
    # Si applica in modo diverso a seconda del tipo di dato (vedi main.py):
    #  - dati gia' processati/filtrati (header con HIGH_CUT_FREQUENCY_HZ
    #    valorizzato, es. ESM processing-type 'MP'): esclusi se il filtro
    #    originale era piu' stretto di questa soglia;
    #  - dati grezzi non filtrati (campo assente, es. ESM 'CV'): questa
    #    soglia non si applica; il file viene comunque escluso se la sua
    #    frequenza di campionamento e' troppo bassa per le bande configurate.
    min_original_high_cut_hz: float = 40.0

    # --- parametri del filtro Butterworth passa-banda ---
    filter_order: int = 1

    # Preprocessing prima del filtraggio: rimozione del trend lineare
    # (include la media) e taper ai bordi, per evitare transitori/ringing
    # del filtro quando il segnale grezzo ha un offset/trend residuo
    # (tipico dei dati non processati, es. ESM 'CV': 'BASELINE NOT REMOVED').
    baseline_correction: bool = True
    taper_fraction: float = 0.05   # frazione di taper (Tukey) per lato, 0 per disattivarlo

    # --- controllo di plausibilita' fisica (terza scrematura, indipendente
    # dalla banda) ---
    # I dati grezzi non processati (es. ESM 'CV') non hanno controllo
    # qualita' automatico: possono contenere glitch del digitalizzatore
    # con valori assurdi (osservati fino a ~2 milioni di m/s^2 in dataset
    # reali). Si scarta l'intero file se il valore assoluto massimo del
    # segnale GREZZO (prima di qualunque filtro) supera questa soglia.
    # Il valore di default (50 m/s^2 ~ 5.1 g) e' volutamente molto
    # generoso: le accelerazioni al suolo piu' forti mai registrate al
    # mondo restano sotto i ~40 m/s^2 (~4 g, eventi vicini alla faglia),
    # quindi non dovrebbe mai scartare un dato sismico reale, solo i
    # glitch strumentali chiaramente impossibili.
    max_plausible_accel_ms2: float = 30.0

    # --- soglia di attivazione (in valore assoluto, in m/s^2). I dati
    # vengono convertiti internamente in m/s^2 (dall'unita' originale
    # dichiarata nell'header UNITS, tipicamente cm/s^2) subito dopo la
    # lettura del file, quindi questa soglia e tutti i risultati/grafici
    # sono sempre espressi in m/s^2. Usata in due punti diversi:
    #  - in main.py: per individuare l'istante di superamento soglia
    #    (analisi "a soglia fissata", usata anche per verificare a
    #    posteriori una soglia calcolata con threshold_calibration.py);
    #  - in threshold_calibration.py: solo come filtro di inclusione
    #    nella popolazione statistica (si escludono le storie che non la
    #    superano mai, cioe' troppo deboli per essere rilevanti) — non
    #    interviene nel calcolo della soglia raccomandata in output.
    activation_threshold: float = 0.05

    # --- le due bande di analisi richieste ---
    bands: Tuple[FilterBand, ...] = field(
        default_factory=lambda: (
            FilterBand(name="20Hz", low_hz=0.1, high_hz=20.0, color="tab:blue"),
            FilterBand(name="40Hz", low_hz=0.1, high_hz=40.0, color="tab:red"),
        )
    )

    # --- linee verticali di riferimento nello scatter plot (delta_t) ---
    # Entrambe sono indipendenti e possono coesistere sullo stesso grafico
    # (una per banda ciascuna). Per disattivarne una, impostare il relativo
    # parametro a -1 (valore fittizio: nessun tempo/percentile e' -1, quindi
    # e' un modo sicuro per dire "non disegnare questa linea").
    #
    # 1) Linea a percentile fisso: calcola il valore di delta_t sotto il
    #    quale cade la percentuale scelta di storie (es. con 5, il 95%
    #    delle storie ha delta_t maggiore del valore trovato).
    delta_t_percentile: float = 5.0     # -1 per disattivare questa linea

    # 2) Linea a tempo fisso: al contrario della precedente, qui si fissa
    #    *il tempo* (in secondi) e si calcola quale percentuale di storie
    #    ha delta_t maggiore di quel tempo.
    delta_t_fixed_threshold_s: float = 1.0   # -1 per disattivare questa linea

    # 3) Seconda soglia (sul picco di accelerazione, in m/s^2), usata SOLO
    #    per il calcolo delle due linee sopra (non filtra lo scatter: tutti
    #    i punti restano visibili nel grafico). Serve a escludere dalle
    #    statistiche delle linee le storie con accelerazione di picco molto
    #    bassa/vicina al rumore (es. stazioni deboli/lontane che superano
    #    di poco activation_threshold), che altrimenti possono dominare il
    #    calcolo del percentile con delta_t quasi nulli non rappresentativi
    #    dell'evento sismico "vero". Con 0 (default) non filtra nulla:
    #    tutte le storie della banda vengono usate per le linee, come nelle
    #    versioni precedenti.
    line_min_peak_accel_ms2: float = 0.5

    # --- linea sottile che collega, nello scatter, il pallino di una
    # banda con quello dell'altra per la STESSA storia temporale (solo
    # per le storie attivate in tutte le bande configurate, non per le
    # "orfane"). Disattivata di default: su popolazioni grandi puo'
    # affollare il grafico: attivarla con --connect-bands o impostando
    # questo flag a True.
    connect_band_points: bool = False

    # --- fasce di PGA per l'analisi dei ritardi medi e delle discordanze
    # stratificate (band_comparison.py, band_mismatch_from_csv.py, replot.py) ---
    # Bordi di fascia (non sovrapposte, [bordo[i], bordo[i+1]), l'ultima
    # aperta verso l'alto) usati per due analisi separate:
    #  1) il ritardo medio (delta_t) per banda, sulle storie con picco
    #     nella fascia;
    #  2) le discordanze tra bande, contate per fascia (una storia con
    #     picco 0.6 conta SOLO nella fascia che lo contiene, es. [0.5,1),
    #     non anche in [0.05,0.1) o altre).
    pga_bins_ms2: Tuple[float, ...] = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0)

    # --- percentile usato per il limite destro dell'asse X nello scatter
    # plot (vedi plotting.py): con 80 il grafico mostra l'80% dei punti
    # (calcolato su TUTTE le bande insieme). Alzarlo (es. 95-99) "allarga"
    # il grafico verso destra, utile per vedere quanto la coda di una
    # banda sborda oltre l'altra.
    x_axis_clip_percentile: float = 95.0

    # --- parametri di seismic_analysis/threshold_calibration.py ---
    # Calcola la soglia di attivazione raccomandata (in m/s^2) tale da
    # garantire un margine minimo tra superamento soglia e picco su una
    # percentuale scelta di sismi, per una localita' e un raggio dati.
    # site_lat/site_lon/radius_km non hanno un default sensato: vanno
    # sempre forniti esplicitamente da riga di comando per questa analisi
    # (obbligatori, vedi threshold_calibration.py --help).
    site_lat: Optional[float] = None
    site_lon: Optional[float] = None
    radius_km: Optional[float] = None

    # Tempo minimo [s] richiesto tra il momento in cui idealmente si
    # sceglie di attivare (superamento soglia) e il picco di accelerazione.
    min_lead_time_s: float = 1.5

    # Percentuale di sismi (sulla popolazione considerata, vedi sopra) che
    # si vuole coprire con il margine minimo richiesto (0-1).
    target_coverage: float = 0.95
