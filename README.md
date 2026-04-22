# KAN Models

Repository per esperimenti con Kolmogorov-Arnold Networks su tre famiglie di modelli:

- `conic`
- `credit_default`
- `stroke`

L'idea del progetto e' semplice:

- i dataset restano in `dataset/`
- i parametri stanno nei file `TOML` dentro `configs/`
- il codice vive in `src/`
- ogni modello ha i suoi entrypoint dentro `src/kan_models/models/...`

## Struttura

Le cartelle principali sono:

```text
dataset/
configs/
  conic/
  credit_default/
  stroke/
src/
  kan_models/
    common/
    models/
      conic/
      credit_default/
      stroke/
artifacts/
```

### Cartelle importanti

- `dataset/`
  Contiene i file dati usati dagli esperimenti.

- `configs/`
  Contiene le configurazioni dei modelli. Qui si cambiano dataset, split, training, pruning e output.

- `src/kan_models/common/`
  Codice condiviso tra piu' modelli.
  Qui trovi:
  - path comuni
  - runtime helpers
  - utility generiche
  - pipeline tabellare condivisa tra `credit_default` e `stroke`

- `src/kan_models/models/`
  Contiene i modelli veri e propri, organizzati per dataset/famiglia.

## Modelli disponibili

### Conic

Cartella:

- `src/kan_models/models/conic/`

Varianti:

- `main.py`
  baseline standard sul dataset conic
- `pruning.py`
  versione pruning
- `continual/main.py`
  continual learning

Config:

- `configs/conic/baseline.toml`
- `configs/conic/pruning.toml`
- `configs/conic/continual.toml`
- `configs/conic/continual_reversed.toml`

### Credit Default

Cartella:

- `src/kan_models/models/credit_default/`

Entry point:

- `main.py`

Config:

- `configs/credit_default/default.toml`

### Stroke

Cartella:

- `src/kan_models/models/stroke/`

Entry point:

- `main.py`
- `pruning.py`

Config attuale:

- `configs/stroke/pruning.toml`

Nota:
al momento `stroke/main.py` e `stroke/pruning.py` usano la stessa configurazione di pruning. Se vuoi una variante standard senza pruning, basta aggiungere un nuovo TOML dedicato.

## Come usare il progetto

Lavora dalla root del repository.

Puoi usare l'ambiente virtuale locale:

```bash
source .venv/bin/activate
```

oppure lanciare direttamente con:

```bash
.venv/bin/python ...
```

## Comandi principali

### Conic baseline

```bash
.venv/bin/python src/kan_models/models/conic/main.py
```

### Conic pruning

```bash
.venv/bin/python src/kan_models/models/conic/pruning.py
```

### Conic continual

```bash
.venv/bin/python src/kan_models/models/conic/continual/main.py
```

### Conic continual reversed

```bash
.venv/bin/python src/kan_models/models/conic/continual/main.py --config configs/conic/continual_reversed.toml
```

### Credit default

```bash
.venv/bin/python src/kan_models/models/credit_default/main.py
```

### Stroke

```bash
.venv/bin/python src/kan_models/models/stroke/main.py
```

### Stroke pruning

```bash
.venv/bin/python src/kan_models/models/stroke/pruning.py
```

## Come cambiare i parametri

La regola pratica e':

1. scegli il modello
2. apri il TOML relativo
3. modifica i parametri
4. rilancia il relativo `main.py`

Le sezioni piu' importanti nei TOML sono:

- `[data]`
  dataset, target, colonne
- `[split]`
  train/validation/test
- `[model]`
  architettura KAN
- `[training]`
  epoche, optimizer, learning rate, patience
- `[pruning]`
  solo per esperimenti pruning
- `[output]`
  file e cartelle di output

Esempio:

```bash
.venv/bin/python src/kan_models/models/credit_default/main.py --config configs/credit_default/default.toml
```

## Dove finiscono gli output

Dipende dal blocco `[output]` del TOML.

In generale:

- `conic` salva metriche e plot nei path definiti nei file `configs/conic/*.toml`
- `credit_default` salva in `artifacts/credit_default_kan/`
- `stroke` salva in `artifacts/stroke_pruning_kan/`

## Nota su `config.py` e file TOML

I file `TOML` contengono i valori della configurazione.

I file `config.py` servono invece a:

- leggere i TOML
- validare i campi
- trasformare la configurazione in oggetti Python strutturati

Quindi:

- `configs/.../*.toml` = parametri dell'esperimento
- `src/.../config.py` = codice che interpreta quei parametri

## In pratica

Se vuoi usare il progetto senza perderti:

- entra nella cartella del modello che ti interessa in `src/kan_models/models/`
- guarda il `main.py`
- apri il TOML associato in `configs/`
- modifica solo la config
- esegui il comando corrispondente

Questa e' la via standard per lavorare sul repository.
