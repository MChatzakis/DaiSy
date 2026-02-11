# Come esportare i diagrammi PlantUML da VSCode/Cursor

Questo progetto include diagrammi PlantUML in `docs/`:
- **`daisy-architecture.puml`** – Architettura dettagliata (classi, package, relazioni)
- **`daisy-components.puml`** – Vista componenti ad alto livello

## 1. Installare l’estensione PlantUML

1. Apri **VSCode** (o **Cursor**).
2. Vai su **Estensioni** (icona a sinistra o `Cmd+Shift+X` su macOS).
3. Cerca **"PlantUML"** (autore: jebbs).
4. Clicca **Installa** su **PlantUML** di jebbs.

## 2. Anteprima del diagramma

1. Apri un file `.puml` (es. `docs/daisy-architecture.puml`).
2. Usa uno di questi modi per aprire l’anteprima:
   - **`Option+D`** (macOS) o **`Alt+D`** (Windows/Linux)
   - **Command Palette** (`Cmd+Shift+P`) → digita **"PlantUML: Preview Current Diagram"** e invio
   - Tasto destro nel file → **"Preview Current Diagram"**

Si aprirà un pannello con il diagramma renderizzato. Aggiorna l’anteprima con **"PlantUML: Preview Current Diagram"** dopo aver modificato il file.

## 3. Esportare in PNG, SVG o altro

### Metodo A – Da Command Palette (consigliato)

1. Apri il file `.puml`.
2. `Cmd+Shift+P` (o `Ctrl+Shift+P`) → **"PlantUML: Export Current Diagram"**.
3. Scegli la cartella di destinazione e il **formato**:
   - **PNG** – immagine raster, adatto a documenti e slide.
   - **SVG** – vettoriale, si ridimensiona senza perdita di qualità.
   - **PDF** – per stampa o documenti PDF.

Il file esportato avrà lo stesso nome del file `.puml` con estensione cambiata (es. `daisy-architecture.png`).

### Metodo B – Tasto destro

1. Apri il file `.puml`.
2. Tasto destro nell’editor → **"Export Current Diagram"**.
3. Scegli percorso e formato come sopra.

### Metodo C – Export con percorso specifico

1. `Cmd+Shift+P` → **"PlantUML: Export Current Diagram"**.
2. Nella finestra "Save As", scegli la cartella (es. `docs/` o il desktop) e nel campo **"Format"** seleziona PNG, SVG o PDF.

## 4. Requisito: Java e Graphviz (per il rendering)

PlantUML usa **Java** e, per molti diagrammi, **Graphviz** (per il layout).

- **Java**: installa un JRE 8+ (es. [Adoptium](https://adoptium.net/) o OpenJDK). Verifica con `java -version`.
- **Graphviz**:  
  - **macOS**: `brew install graphviz`  
  - **Ubuntu/Debian**: `sudo apt install graphviz`  
  - **Windows**: scarica da [graphviz.org](https://graphviz.org/download/).

Se l’anteprima non si apre o l’export fallisce, controlla che Java e Graphviz siano nel `PATH` e riavvia l’editor.

## 5. Riepilogo comandi utili

| Azione                    | Comando / scorciatoia      |
|---------------------------|----------------------------|
| Anteprima diagramma       | `Option+D` (Mac) / `Alt+D` (Win/Linux) |
| Export diagramma          | Command Palette → "PlantUML: Export Current Diagram" |
| Aggiornare anteprima      | "PlantUML: Preview Current Diagram" |

Dopo l’export, i file (es. `daisy-architecture.png`) possono essere inseriti in README, wiki o documentazione del progetto.
