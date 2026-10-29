# Analisi storie temporali di accelerazione (formato DYNA/ESM)

Pipeline Python per l'analisi di storie temporali di accelerazione sismica
in formato DYNA (ESM - Engineering Strong Motion database).

## Cosa fa

1. **Prima scrematura**: esclude tutti i file relativi alla componente
   verticale (`Z`), riconosciuta dal nome file (gruppo `HN?`) e verificata
   incrociando il campo header `STREAM`.
2. **Seconda scrematura**: controllo di plausibilità fisica sul dato
   grezzo — protegge da glitch del digitalizzatore (osservati in dataset
   reali fino a ~2 milioni di m/s², tipici dei dati non processati come
   ESM `CV`, privi di controllo qualità automatico): si scarta il file
   se il valore assoluto massimo del segnale grezzo supera una soglia
   fisicamente plausibile (default `50` m/s² ≈ 5.1 g — molto generosa,
   ben oltre le accelerazioni al suolo più forti mai registrate al mondo,
   così da non scartare mai un dato sismico reale).
3. **Terza scrematura**: esclude i file la cui registrazione originale è
   già stata filtrata (in fase di processing ESM) con una
   `HIGH_CUT_FREQUENCY_HZ` inferiore a una soglia minima (default 40 Hz);
   sui dati grezzi (campo assente) verifica invece che la frequenza di
   campionamento sia sufficiente per le bande richieste.
4. **Analisi**, per ogni file rimasto:
   - filtra la storia temporale con due filtri Butterworth passa-banda a
     fase nulla (`scipy.signal.sosfiltfilt`), di default nelle bande
     `[0.1–20] Hz` e `[0.1–40] Hz`, ordine 1 (parametrizzabile);
   - individua il primo istante in cui il valore assoluto del segnale
     filtrato supera la soglia di attivazione (default `0.05`);
   - individua l'istante e il valore dell'accelerazione massima in valore
     assoluto — **calcolati sempre**, anche quando la soglia non viene
     superata (a differenza delle versioni precedenti, che in quel caso
     scartavano il picco senza salvarlo);
   - calcola il tempo trascorso tra superamento soglia e picco (`delta_t`),
     quando la soglia è stata superata.
5. **Output**:
   - `output/results.csv`: una riga per ogni combinazione file × banda,
     **sempre** (anche se la soglia non è stata superata), con una colonna
     `activated` che lo indica; `t_cross_s`/`delta_t_s` sono vuoti quando
     `activated=False`, ma il picco (`peak_abs_accel_ms2`) è comunque
     presente;
   - `output/band_summary.csv`: per ciascuna banda, quante storie hanno
     superato la soglia di attivazione (`activated=True`) sul totale delle
     storie candidate (cioè quelle sopravvissute alle scremature);
   - `output/exclusion_log.csv`: motivo di esclusione **strutturale** di
     ogni file scartato nelle varie fasi (componente Z, plausibilità
     fisica, idoneità al filtraggio) — non include più il caso "soglia mai
     superata", che ora è una riga normale in `results.csv` con
     `activated=False`;
   - `output/scatter_delta_t_vs_peak.png`: grafico a dispersione (solo
     storie con `activated=True`, dato che il concetto di `delta_t` ha
     senso solo in quel caso) con
     `X = delta_t [s]`, `Y = accelerazione massima assoluta [m/s²]`,
     colore diverso per le due bande (blu = 20 Hz, rosso = 40 Hz). Per
     ciascuna banda possono comparire fino a due linee verticali di
     riferimento indipendenti (vedi sotto).
6. **Confronto tra le due bande** (richiede esattamente 2 bande
   configurate): per ogni storia, si verifica se l'attivazione è concorde
   o discorde tra le due bande (es. attivata a 40Hz ma non a 20Hz) — vedi
   la sezione dedicata più sotto.

### Linee verticali di riferimento sul grafico

Per ciascuna banda si possono mostrare fino a due linee verticali
tratteggiate, ciascuna attivabile/disattivabile e parametrizzabile
indipendentemente (parametri in `config.py`, sovrascrivibili da CLI):

1. **Percentile fisso** (`Config.delta_t_percentile`, default `5`,
   CLI `--delta-t-percentile`): si sceglie una percentuale e si calcola
   il valore di delta_t sotto il quale cade quella percentuale di storie
   — con `5`, il 95% delle storie ha delta_t maggiore del valore trovato
   (linea tratteggiata `--`).
2. **Tempo fisso** (`Config.delta_t_fixed_threshold_s`, default `1.0`,
   CLI `--delta-t-fixed-threshold`): all'opposto, si fissa il tempo (in
   secondi) e si calcola quale percentuale di storie ha delta_t maggiore
   di quel tempo (linea punteggiata `:`).

Le due linee sono indipendenti e possono coesistere sullo stesso grafico.
**Per disattivarne una, impostare il relativo parametro a `-1`** (valore
fittizio, dato che né un percentile né un tempo possono essere negativi
nella realtà):

```bash
# Solo la linea a tempo fisso (7s), percentile disattivato
python -m seismic_analysis.main --delta-t-percentile -1 --delta-t-fixed-threshold 7

# Nessuna linea verticale
python -m seismic_analysis.main --delta-t-percentile -1 --delta-t-fixed-threshold -1
```

### Seconda soglia: filtra solo le storie usate per calcolare le linee

Le due linee sopra possono essere calcolate su un **sottoinsieme** delle
storie della banda, invece che su tutte: `Config.line_min_peak_accel_ms2`
(default `0.5`, CLI `--line-min-peak`) è una seconda soglia, in m/s²,
sull'**accelerazione di picco** — indipendente da `activation_threshold`.
Solo le storie con picco ≥ questa soglia contribuiscono al calcolo delle
due linee; lo **scatter continua a mostrare tutti i punti** della banda,
filtro o non filtro (la seconda soglia non nasconde nessun punto, incide
solo sulle statistiche delle linee).

Serve perché, su popolazioni molto grandi ed eterogenee (es. migliaia di
stazioni a distanze molto diverse dall'epicentro), le storie più deboli —
che superano di poco `activation_threshold` per via del rumore, non per
un vero scuotimento significativo — possono avere delta_t vicinissimi a
zero (il superamento soglia e il "picco" coincidono quasi, semplicemente
perché il segnale oscilla appena sopra/sotto la soglia): se sono
abbastanza numerose, dominano il calcolo del percentile, spostando la
linea verso valori poco rappresentativi dell'evento sismico "vero" (es.
`P5 = 0.01s` invece di un valore più significativo). Impostando una
seconda soglia più alta si escludono queste storie marginali dal solo
calcolo delle linee, senza toccare il resto dell'analisi:

```bash
python -m seismic_analysis.main --line-min-peak 0.3
```

La legenda riporta sempre quante storie sono state effettivamente usate
per le linee (es. `[su 3302/13133 storie con picco >= 0.3 m/s²]`), e se
nessuna storia della banda supera la seconda soglia le linee di quella
banda non vengono disegnate (segnalato esplicitamente in legenda) invece
di fallire silenziosamente.

### Limiti degli assi del grafico

Per leggibilità, gli assi dello scatter plot sono limitati:
- **X** (delta_t): parte sempre da 0 (eventuali delta_t negativi non sono
  mostrati) ed è tagliato a destra al percentile scelto in
  `Config.x_axis_clip_percentile` (default **95**, CLI
  `--x-axis-clip-percentile`, con un piccolo margine): il grafico mostra
  quindi il 95% dei punti, tagliando solo la coda più estrema. Alzarlo
  ulteriormente (es. 99) "allarga" il grafico verso destra — utile per
  vedere quanto la coda di una banda sborda oltre l'altra.
- **Y** (accelerazione di picco): tagliato con la regola di Tukey
  (Q3 + 1.5×IQR), lo standard usato nei boxplot per individuare gli
  outlier — robusto anche quando gli outlier sono più del 5% dei punti.

In entrambi i casi i punti fuori dai limiti non vengono mostrati nel
grafico ma restano nei CSV (`results.csv`), e le percentuali/linee (vedi
sopra) sono sempre calcolate sui dati completi, non sulla sola porzione
visibile. Se il grafico sembra "sbagliato" (es. una linea di percentile
che appare vicina a un bordo), è spesso perché senza un limite sull'asse
alcuni outlier estremi comprimono visivamente tutto il resto: ecco perché
entrambi gli assi sono limitati per default.

Il riepilogo dei conteggi per banda viene anche stampato a schermo al
termine dell'esecuzione.

### Rigenerare il grafico senza rieseguire l'analisi

`output/results.csv` contiene già tutto ciò che serve per disegnare il
grafico (`delta_t_s`, `peak_abs_accel_ms2`, banda, frequenze) — cambiare
solo i parametri delle linee verticali (percentile, tempo fisso, seconda
soglia) o i limiti degli assi non richiede rifiltrare le storie
temporali. `seismic_analysis/replot.py` sfrutta esattamente questo: legge
un `results.csv` già prodotto e rigenera solo il PNG, in pochi secondi
anche su decine di migliaia di righe (contro i minuti che servirebbero a
rieseguire tutto il filtraggio):

```bash
# Si esegue main.py una volta sola...
python -m seismic_analysis.main

# ...poi si può iterare rapidamente su come impostare le linee, senza
# più toccare i dati grezzi:
python -m seismic_analysis.replot --results-csv output/results.csv \
    --output output/scatter_p10.png --delta-t-percentile 10

python -m seismic_analysis.replot --results-csv output/results.csv \
    --output output/scatter_filtered.png --line-min-peak 0.3
```

Parametri disponibili: `--delta-t-percentile`, `--delta-t-fixed-threshold`,
`--line-min-peak`, `--pga-bins`, `--x-axis-clip-percentile` (stesso
significato delle rispettive opzioni di `main.py`), `--output` per il
percorso dello scatter PNG (default: accanto al CSV, con suffisso
`_replot`) e `--output-dir` per gli output aggiuntivi (confronto tra
bande, fasce di PGA — vedi sezione dedicata più sotto), rigenerati
automaticamente quando il CSV ha la colonna `activated`.

**Limite importante**: questo script rilegge solo il CSV, non le storie
temporali originali. Tutto ciò che riguarda il **filtraggio** (ordine del
filtro, bande di frequenza, soglia di attivazione, preprocessing,
scremature, soglia di plausibilità fisica) è già "congelato" nei valori
del CSV e non può essere cambiato qui: per modificare quei parametri
serve rieseguire `main.py` da capo sui dati grezzi.

## Confronto tra bande (attivazioni discordanti)

Con due bande di filtraggio, una storia può attivare in una banda ma non
nell'altra (tipicamente: attiva a 40Hz — banda più larga, più probabile
che ci sia energia sopra soglia da qualche parte nello spettro — ma non a
20Hz). Il progetto offre due strumenti per analizzare queste discordanze:

### 1. Automatico, dentro `main.py` (consigliato per nuove analisi)

Da quando `results.csv` include una riga per **ogni** storia × banda
(anche senza attivazione, con il picco comunque calcolato — vedi sopra),
`main.py` produce automaticamente, ad ogni esecuzione:

- `output/band_comparison.csv`: una riga per storia, con attivazione e
  picco di **entrambe** le bande affiancati (`attivata_20Hz`,
  `attivata_40Hz`, `picco_20Hz_ms2`, `picco_40Hz_ms2`, ...) e un flag
  `discordanza`.
- `output/band_mismatch.csv`: solo le righe con `discordanza=True`.
- `output/band_comparison.png`: due pannelli —
  - a sinistra, uno scatter (scala log) del picco nella banda A contro il
    picco nella banda B per **tutte** le storie, colorato per categoria
    (concordi entrambe attivate, concordi nessuna, discordi in un senso o
    nell'altro), con le linee della soglia di attivazione;
  - a destra, per le sole storie discordanti, l'istogramma del picco
    raggiunto nella banda **che ha attivato** confrontato con quello
    raggiunto nella banda **che non ha attivato** — la risposta diretta
    alla domanda "quale picco è stato raggiunto quando le due bande sono
    in disaccordo".
- `output/delay_by_pga_band.csv/png`: ritardo medio (delta_t) per banda,
  per fascia di picco (PGA) — fasce configurabili in `Config.pga_bins_ms2`
  (default `0.05, 0.1, 0.2, 0.5, 1, 2, 4` m/s², CLI `--pga-bins`). Il grafico è un
  bar chart a barre affiancate (una per banda), con barre d'errore
  (deviazione standard) e il numero di storie per fascia.
- `output/mismatch_by_pga_band.csv/png`: percentuale di storie
  discordanti tra le bande, **non solo sul totale ma per ciascuna fascia
  di PGA** nello stesso elenco `pga_bins_ms2`, con le stesse fasce **non
  sovrapposte** di `delay_by_pga_band` (es. "tra le storie con picco in
  [1, 2) m/s², quante sono discordanti?") — ogni storia conta in una sola
  fascia, mai in più di una. Il grafico mostra il numero di storie in
  ciascuna fascia (barre) e la percentuale di discordanti (linea rossa,
  asse secondario).

Nello scatter principale (`scatter_delta_t_vs_peak.png`), inoltre, le
storie attivate in una banda ma **senza corrispettivo attivato nelle
altre bande** vengono disegnate in **nero** invece che nel colore della
banda, così si distinguono a colpo d'occhio le attivazioni concordi da
quelle discordanti — restano comunque conteggiate nelle statistiche delle
linee verticali, solo il colore cambia. Con `Config.connect_band_points`
(default `False`, CLI `--connect-bands`) si disegna anche una linea
sottile e semitrasparente che collega il pallino di una banda con quello
dell'altra per la stessa storia (solo per le storie attivate in **tutte**
le bande — le orfane non hanno nulla da collegare): utile per vedere a
colpo d'occhio quanto si sposta lo stesso evento tra bande diverse.
Disattivato di default perché su popolazioni grandi può affollare il
grafico.

Questa è la versione "completa": avendo sempre il picco per entrambe le
bande, non ci sono limitazioni sui dati disponibili.

### 2. Da CSV già esistenti (senza rieseguire l'analisi)

Se hai già un `results.csv` prodotto dalla versione **aggiornata** del
programma (con la colonna `activated`), `replot.py` rigenera anche tutte
queste analisi partendo dal solo CSV (vedi sezione precedente
"Rigenerare il grafico senza rieseguire l'analisi", incluso il flag
`--connect-bands`) — è il caso più comune, dato che il CSV contiene già
tutto il necessario.

Se invece hai un `results.csv` + `exclusion_log.csv` prodotti da una
versione **precedente** del programma (quella che scartava le storie non
attivate senza salvarne il picco), `seismic_analysis/band_mismatch_from_csv.py`
ricostruisce comunque le discordanze — e, dalla stessa versione, anche il
ritardo medio per fascia di PGA e le discordanze per fascia — da questi
soli file:

```bash
python -m seismic_analysis.band_mismatch_from_csv \
    --results-csv output/results.csv \
    --exclusion-csv output/exclusion_log.csv \
    --output-dir output \
    --pga-bins 0.05 0.1 0.2 0.5 1 2 4
```

Produce `band_mismatch_from_csv.csv/png`, `delay_by_pga_band_from_csv.csv/png`
e `mismatch_by_pga_band_from_csv.csv/png`. **Limite**: da questi CSV
è recuperabile solo il picco della banda che **ha** attivato — quello
della banda che non ha attivato non è mai stato salvato dalla versione
precedente del programma, quindi non è disponibile in nessun modo senza
rieseguire l'analisi con la versione aggiornata di `main.py`. Per lo
stesso motivo, lo scatter principale rigenerato da questi CSV con
`replot.py` non può disegnare i pallini neri (richiede sapere se il file
è assente nell'altra banda, informazione che con questo formato di CSV
è comunque ricostruibile dalla sola presenza/assenza delle righe, e
quindi funziona correttamente anche qui).

## Struttura del progetto

```
seismic_project/
├── output/                     # risultati generati (CSV + grafico)
├── requirements.txt
└── seismic_analysis/
    ├── config.py                    # tutti i parametri configurabili (incluso il percorso dati sul NAS)
    ├── io_dyna.py                    # parsing dei file DYNA (header + dati + coordinate)
    ├── geo.py                        # distanza geografica (Haversine)
    ├── screening.py                  # scrematura condivisa (componente Z, idoneita' al filtraggio)
    ├── filtering.py                  # preprocessing (demean/detrend/taper) + filtro Butterworth
    ├── analysis.py                   # superamento soglia + ricerca picco (usato da main.py)
    ├── plotting.py                   # scatter plot di main.py
    ├── download_data.py              # download automatico da ISIDe/ESM
    ├── main.py                       # pipeline "a soglia fissata" (CLI)
    ├── band_comparison.py            # confronto attivazione tra bande (usato da main.py)
    ├── band_mismatch_from_csv.py     # stesso confronto, da CSV già esistenti (CLI)
    ├── threshold_calibration.py      # calibrazione "requirement-driven" della soglia (CLI)
    ├── calibration_plotting.py       # grafico copertura-vs-soglia di threshold_calibration.py
    └── replot.py                     # rigenera il grafico da un results.csv esistente (CLI)
```

**Nota sulla cartella dati**: a differenza di versioni precedenti, la
cartella con i file DYNA (`.txt`/`.ASC`) da analizzare **non fa più parte
del progetto**: si trova in una cartella condivisa sul NAS, configurata
in `seismic_analysis/config.py` (`Config.input_dir`):

```
Z:\01_SVILUPPO PRODOTTO\26_09_ACT_LOGIC\01_DATI\Terremoti scaricati\data
```

Sia `main.py` (lettura per l'analisi) sia `download_data.py` (scrittura
dei file scaricati) puntano di default a questo percorso: è sufficiente
avere l'unità di rete `Z:` mappata. Per usare una cartella diversa
(es. per un test locale), passare esplicitamente `--input-dir` /
`--output-dir` a riga di comando, oppure modificare il valore di default
in `config.py`.

## Installazione

```bash
pip install -r requirements.txt
```

## Utilizzo

Esecuzione con i parametri di default (legge dalla cartella dati sul NAS,
scrive in `output/`):

```bash
python -m seismic_analysis.main
```

Parametri disponibili (tutti opzionali, sovrascrivono `config.py`):

```bash
python -m seismic_analysis.main \
    --input-dir "Z:\01_SVILUPPO PRODOTTO\26_09_ACT_LOGIC\01_DATI\Terremoti scaricati\data" \
    --output-dir output \
    --order 1 \
    --threshold 0.05 \
    --min-high-cut 40 \
    --band1 0.1 20 \
    --band2 0.1 40
```

- `--order`: ordine del filtro Butterworth (applicato a entrambe le bande).
- `--threshold`: soglia di attivazione, in valore assoluto, stessa unità
  di misura del dato (`cm/s²` secondo l'header DYNA).
- `--min-high-cut`: soglia minima di `HIGH_CUT_FREQUENCY_HZ` originale
  sotto la quale un file viene scartato nella seconda scrematura.
- `--max-plausible-accel`: soglia [m/s²] di plausibilità fisica sul dato
  grezzo (default 30.0 ≈ 3.1 g); file con valori sopra vengono scartati
  come probabili glitch del digitalizzatore.
- `--band1` / `--band2`: frequenze di taglio [Hz] (basso, alto) delle due
  bande di filtraggio.
- `--delta-t-percentile` / `--delta-t-fixed-threshold`: parametri delle
  due linee verticali di riferimento nel grafico (vedi sezione dedicata
  più sotto); impostare a `-1` per disattivarne una.
- `--line-min-peak`: seconda soglia [m/s²] sull'accelerazione di picco
  che filtra solo il calcolo delle due linee (non lo scatter); vedi
  sezione dedicata più sotto.
- `--no-baseline-correction` / `--taper-fraction`: controllano il
  preprocessing prima del filtraggio (vedi sezione dedicata più sotto).
- `--pga-bins`: bordi delle fasce di PGA [m/s²] per il ritardo medio per
  fascia e le discordanze per soglia (default `0.05 0.1 0.2 0.5 1 2 4`); vedi
  sezione "Confronto tra bande" più sotto.
- `--x-axis-clip-percentile`: percentile di delta_t per il limite destro
  dell'asse X nello scatter (default 95); vedi "Limiti degli assi".
- `--connect-bands`: disegna una linea sottile che collega, nello
  scatter, il pallino di una banda con quello dell'altra per la stessa
  storia (disattivato di default); vedi "Confronto tra bande".

Per modifiche più strutturali (es. aggiungere una terza banda, cambiare i
colori del grafico) è sufficiente editare `seismic_analysis/config.py`.

## Calibrazione della soglia di attivazione (strategia "requirement-driven")

Oltre alla pipeline `main.py` (che parte da una soglia **fissata** e ne
verifica empiricamente il comportamento — la strategia "bottom-up"),
il progetto include `seismic_analysis/threshold_calibration.py`, che
implementa la strategia opposta ("top-down"): a partire da un
**requisito** (margine minimo richiesto + copertura desiderata),
**calcola quale soglia** lo soddisfa.

### Logica

Per ogni storia temporale (dopo le stesse scremature di `main.py`, più un
filtro geografico sulla distanza tra la **stazione** che l'ha registrata
e la località di interesse):

1. Si individua `t_peak`: l'istante dell'accelerazione massima assoluta
   sull'intera traccia filtrata.
2. Si arretra di `min_lead_time` secondi da `t_peak`, ottenendo l'istante
   limite `t_check = t_peak - min_lead_time`.
3. Si calcola `tau_i = max(|segnale|)` sull'intervallo `[0, t_check]`:
   è il valore di soglia più alto che, se scelto come soglia operativa,
   farebbe comunque scattare l'attivazione entro `t_check` — quindi con
   almeno `min_lead_time` secondi di margine dal picco — per **quella
   specifica storia**. Se `t_check` cade prima dell'inizio della traccia
   (il picco è troppo vicino all'inizio della registrazione), il margine
   richiesto non è raggiungibile con **nessuna** soglia per quella
   storia: viene marcata come "non raggiungibile".
4. Sull'insieme di tutti i `tau_i` della popolazione, la **soglia
   raccomandata** è il valore più alto per cui almeno la percentuale di
   copertura richiesta ha `tau_i >= soglia`.

La soglia storica (`Config.activation_threshold`, es. `0.05` m/s²) resta
usata **solo come filtro di ingresso** nella popolazione statistica: le
storie che non la superano mai (troppo deboli per essere rilevanti)
vengono scartate a monte e non influenzano il calcolo della soglia
raccomandata.

### Uso

```bash
python -m seismic_analysis.threshold_calibration \
    --site-lat 42.83 --site-lon 13.11 --radius-km 50 \
    --min-lead-time 1.5 --coverage 0.95
```

Parametri obbligatori:
- `--site-lat` / `--site-lon`: coordinate della località di interesse
  (gradi decimali).
- `--radius-km`: raggio entro cui tenere le stazioni — oltre, la storia
  temporale viene scartata (distanza calcolata dalla **stazione**, non
  dall'epicentro: sono le caratteristiche di sito della stazione a
  determinare se la registrazione è rappresentativa dello scuotimento
  atteso nella località di interesse).

Parametri opzionali principali:
- `--min-lead-time`: margine minimo richiesto in secondi (default `1.5`).
- `--coverage`: percentuale di copertura richiesta, 0-1 (default `0.95`).
- `--activation-threshold`: soglia storica usata come filtro di ingresso
  (default `0.05` m/s², vedi sopra).
- Tutti i parametri di filtraggio di `main.py` (`--order`, `--band1`,
  `--band2`, `--min-high-cut`, `--no-baseline-correction`,
  `--taper-fraction`) sono disponibili anche qui, con lo stesso significato.

### Output

- `output/calibration_summary.csv`: per ciascuna banda, popolazione
  considerata, numero di storie con margine non raggiungibile, copertura
  massima raggiungibile, e la **soglia raccomandata** (o l'indicazione
  che l'obiettivo non è raggiungibile con nessuna soglia).
- `output/calibration_records.csv`: dettaglio per ogni storia × banda
  (magnitudo, distanza dalla stazione, `t_peak`, `tau_i`, se inclusa
  nella popolazione) — traccia di controllo completa, utile anche in
  ottica di audit/certificazione.
- `output/calibration_exclusion_log.csv`: motivo di esclusione di ogni
  file scartato (componente Z, fuori raggio, coordinate mancanti,
  frequenza di campionamento insufficiente).
- `output/calibration_coverage_vs_threshold.png`: grafico "copertura vs
  soglia" — per ciascuna banda, una curva a gradini che mostra, per ogni
  possibile valore di soglia, quale percentuale della popolazione
  raggiungerebbe il margine minimo richiesto. La soglia raccomandata è
  il punto in cui la curva incrocia la copertura richiesta (linea
  orizzontale tratteggiata); se la curva non arriva mai a quel livello,
  l'obiettivo non è raggiungibile con nessuna soglia positiva.

### Caso "obiettivo non raggiungibile"

Se anche con soglia pari a zero la percentuale di copertura richiesta non
è raggiungibile (troppe storie hanno il picco troppo vicino all'inizio
della registrazione rispetto al margine richiesto), lo strumento lo
segnala esplicitamente invece di restituire un numero fuorviante,
riportando la copertura massima realisticamente ottenibile con questi
dati. È un limite fisico del fenomeno (tipicamente eventi vicini con
rottura impulsiva, dove l'onset e il picco sono quasi simultanei), non
un problema risolvibile stringendo ulteriormente la soglia — va gestito
riconoscendolo esplicitamente come rischio residuo, eventualmente
coperto da un'altra barriera di sicurezza.

### Come usare insieme le due analisi

L'approccio consigliato è "progetta con l'una, verifica con l'altra":
1. Usare `threshold_calibration.py` per derivare la soglia raccomandata
   a partire dal requisito di margine e copertura.
2. Passare quella soglia a `main.py` (`--threshold <valore_calcolato>`)
   per verificarne empiricamente il comportamento sugli stessi dati (o
   su un set indipendente), con lo scatter plot `delta_t` vs picco e le
   linee di riferimento già disponibili in quella pipeline.

## Download automatico dei dati

Il progetto include `seismic_analysis/download_data.py` per scaricare
automaticamente altri terremoti italiani da integrare nella cartella dati
sul NAS, senza bisogno di registrazione (per i dati pubblicamente
rilasciati).

Fonti usate:
- **ISIDe** (INGV, https://terremoti.ingv.it/iside) — il catalogo
  parametrico più esteso disponibile per l'Italia (data, epicentro,
  magnitudo di *tutti* i terremoti strumentali dal 1985 a oggi, anche di
  magnitudo molto bassa). Usato solo per farsi un'idea di quanti/quali
  eventi ci sono — non fornisce forme d'onda.
- **ESM** (Engineering Strong-Motion database, https://esm-db.eu) — la
  stessa fonte da cui provengono i file già usati nel progetto. Copre
  eventi con magnitudo ≥ 4.0 in area euro-mediterranea. Espone le forme
  d'onda a diversi livelli di processing (parametro `--processing-type`):
  - **`CV`** (default) — dato **grezzo**: solo convertito da conteggi
    digitali a cm/s², nessun filtro applicato, nessun ritaglio. Campi
    header `FILTER_TYPE`, `HIGH_CUT_FREQUENCY_HZ`, ecc. vuoti. È il dato
    da scegliere per filtrare autonomamente (come fa questo progetto).
  - `MP` — dato processato manualmente da un analista (correzione di
    linea di base, filtraggio, ritaglio attorno all'evento).
  - `AP` — processato automaticamente, via di mezzo tra CV e MP.

Esempi:

```bash
# Elenca gli eventi italiani (catalogo ISIDe, molto esteso) in un periodo
python -m seismic_analysis.download_data list-events \
    --start 2016-08-01 --end 2016-12-31 --min-magnitude 4.5

# Scarica le forme d'onda ESM GREZZE (default: CV) per gli eventi italiani
# in quel periodo, e le salva nella cartella dati sul NAS (default)
python -m seismic_analysis.download_data download-waveforms \
    --start 2016-08-01 --end 2016-12-31 --min-magnitude 4.5
```

Dopo il download, i nuovi file sono immediatamente analizzabili con
`python -m seismic_analysis.main` (il parser riconosce sia l'estensione
`.txt` dei file originali sia la `.ASC` prodotta dal download automatico).

Opzioni utili di `download-waveforms`:
- `--output-dir`: cartella di destinazione (default: il percorso NAS
  configurato in `config.py`).
- `--network` / `--station`: per limitare il download a una rete o
  stazione specifica (es. `--network IT`).
- `--processing-type {CV,MP,AP}`: livello di processing (vedi sopra).
- `--sleep`: pausa in secondi tra una richiesta e l'altra (default 1s,
  per non sovraccaricare il servizio).
- `--force`: riscarica/sovrascrive anche gli eventi/file già presenti
  in locale (per default vengono saltati, vedi sotto).

Nota: ESM copre solo eventi con M ≥ 4.0 — un `--min-magnitude` inferiore
non produce risultati aggiuntivi in `download-waveforms` (viene comunque
elevato automaticamente a 4.0).

### File già scaricati: non vengono ri-scaricati

Prima di interrogare ESM per un evento, lo script controlla se nella
cartella di destinazione esistono già file il cui nome contiene l'ID di
quell'evento (tutti i file DYNA scaricati da ESM lo includono, es.
`...D.EMSC-20161026_0000077.ACC.CV.ASC`). Se ne trova anche solo uno,
salta del tutto la richiesta HTTP per quell'evento (log `[==] ... gia'
presente, salto`) — comodo per riprendere un download interrotto, o per
allargare progressivamente il periodo/l'area senza riscaricare tutto da
capo. Per forzare comunque un nuovo download (es. dopo aver cambiato
`--processing-type`, che produce file con nome diverso e quindi non
verrebbe comunque rilevato come "già presente"), usare `--force`.

### Dati grezzi (CV) e seconda scrematura

Con `--processing-type CV` il campo `HIGH_CUT_FREQUENCY_HZ` dell'header è
sempre vuoto (non essendoci alcun filtro originale da poter scartare).
La pipeline (`seismic_analysis/main.py`) lo gestisce automaticamente: per
i dati grezzi, la seconda scrematura non si basa più su quel campo, ma
verifica che la **frequenza di campionamento** del file sia sufficiente
per le bande di filtraggio richieste (frequenza di Nyquist = fs/2 deve
superare la frequenza di taglio più alta tra le bande configurate); in
caso contrario il file viene scartato con un motivo esplicito nel log.

**Attenzione: la frequenza di campionamento NON è sempre 200 Hz.** Dipende
dal digitalizzatore della singola stazione/rete: interrogando un evento
con molte stazioni (Norcia, M6.5, 915 tracce) si osservano concretamente
tre valori diversi:

| Frequenza | dt (SAMPLING_INTERVAL_S) | N. tracce osservate |
|---|---|---|
| 200 Hz | 0.005 s | 660 (72%) |
| 100 Hz | 0.010 s | 222 (24%) |
| 125 Hz | 0.008 s | 33 (4%) |

Il parser (`io_dyna.py`) legge sempre il valore corretto dall'header di
ciascun file — non va mai assunto un valore fisso — e la frequenza
effettiva di ogni file (`fs`) viene usata automaticamente nel progettare
i filtri Butterworth. Con dt=0.010s (100 Hz) la banda [0.1–40] Hz resta
appena valida (Nyquist=50 Hz); con dt=0.008s (125 Hz) c'è più margine
(Nyquist=62.5 Hz); una stazione ancora più lenta verrebbe correttamente
scartata dalla scrematura sopra descritta.

## Note

- Il filtraggio è a fase nulla (`sosfiltfilt`): non introduce sfasamento
  temporale, condizione necessaria per una stima corretta del tempo tra
  superamento soglia e picco.
- **Preprocessing prima del filtraggio** (`Config.baseline_correction`,
  default attivo; CLI `--no-baseline-correction` per disattivarlo): prima
  di applicare il passa-banda, il segnale viene ripulito rimuovendo il
  trend lineare (che include la media) e applicando un taper (finestra di
  Tukey, frazione configurabile con `--taper-fraction`, default 0.05) ai
  bordi. Serve a evitare transitori/ringing del filtro quando il segnale
  grezzo ha un offset o un trend residuo ai bordi — tipico dei dati non
  processati (es. ESM `CV`, header `BASELINE NOT REMOVED`). È buona
  pratica standard in sismologia (demean → detrend → taper → filtro).
  **Nota diagnostica**: su un test con 316 tracce reali (evento Norcia
  M6.6), questo preprocessing non ha cambiato in modo apprezzabile la
  distribuzione dei `delta_t` (stessi percentili, stesso numero di casi
  sotto 0.5s): il cluster di storie con `delta_t` molto piccolo osservato
  in precedenza *non* è causato da un transitorio del filtro ai bordi
  (verificato: solo 1 traccia su 316 aveva il picco nel primo 5% della
  registrazione, ed era un picco vero, non spurio) — è invece legato al
  fatto che la soglia di attivazione fissa (0.05 m/s²) è molto vicina
  all'ampiezza del segnale per le stazioni deboli/lontane dall'epicentro,
  indipendentemente dal filtraggio.
- **Unità di misura**: i file DYNA originali sono in `cm/s²` (campo header
  `UNITS`). Subito dopo la lettura, `io_dyna.py` converte automaticamente
  ogni storia temporale in **m/s²**: tutta la pipeline (soglia di
  attivazione, filtri, colonne dei CSV, assi del grafico) lavora quindi in
  m/s². La soglia di attivazione di default è `0.05` m/s² (= 5 cm/s²).
  Se in futuro dovessero arrivare file con `UNITS` diverso da `cm/s^2`
  (es. `m/s^2` o `g`), la conversione viene applicata comunque in modo
  corretto (vedi dizionario `_UNIT_TO_MS2` in `io_dyna.py`); un'unità non
  riconosciuta genera un errore esplicito invece di un valore silenziosamente
  sbagliato.
