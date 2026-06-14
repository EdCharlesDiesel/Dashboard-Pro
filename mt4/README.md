# MT4 auto-export → Trade Journal auto-import

`HistoryAutoExport.mq4` is an Expert Advisor that writes your closed-trade
history **and** current account balance to an HTML file every few minutes. The
file matches the format the app's parser (`src/services/mt4_import.py`) reads, so
the Trade Journal's **🔄 Auto-import (5 min)** tab can pick it up and keep both
your trade history and your Setup Ranker account balance in sync — no manual
uploads.

## Install

1. In MT4: **File → Open Data Folder**. Copy `HistoryAutoExport.mq4` into
   `MQL4\Experts\`.
2. In MT4: **Navigator → Expert Advisors → right-click → Refresh** (or open the
   file in MetaEditor and press **Compile / F7**).
3. Drag **HistoryAutoExport** onto any chart (the symbol doesn't matter).
4. In the dialog, tick **Allow live trading** (the EA only writes a file — no DLLs
   needed), then **OK**. Make sure the toolbar **AutoTrading** button is green.
5. A smiley face appears top-right of the chart and a `Comment` overlay shows the
   last write time and balance.

## Inputs

| Input | Default | Meaning |
|---|---|---|
| `ExportFileName` | `history.htm` | Output file name |
| `RefreshSeconds` | `300` | Export interval (300 = 5 min) |
| `UseCommonFolder` | `true` | Write to the shared `Common\Files` folder (path is stable across terminals) |

## Where the file lands → what to put in the app

MT4 sandboxes file writes, so the file goes to one of:

- **Common folder** (`UseCommonFolder = true`, default):
  `C:\Users\<you>\AppData\Roaming\MetaQuotes\Terminal\Common\Files\history.htm`
- **Per-terminal folder** (`UseCommonFolder = false`): open **File → Open Data
  Folder**, then `MQL4\Files\history.htm`.

Copy that full path into the Trade Journal → **Import → 🔄 Auto-import (5 min)** →
*MT4 report file path*, set your **Broker server UTC offset**, and flip
**Auto-import every 5 minutes** on. The app re-reads the file whenever its
modified time changes, imports new closed trades (deduped by ticket), and updates
the live account balance.

## Important

- **Load full history first.** The EA only exports what's in MT4's **Account
  History** tab. Right-click that tab → **All History** (or a wide custom period)
  so the export isn't limited to the last few days.
- **Keep MT4 running.** EAs only fire while the terminal is open with the EA
  attached to a chart.
- **Broker time.** Timestamps are written in broker server time (often UTC+2/+3).
  Set the offset in the app so sessions bucket correctly — the EA doesn't convert.
- **Running the app in Docker?** Mount the export folder into the container and
  point the path field at the container path, e.g.:
  `-v "C:/Users/<you>/AppData/Roaming/MetaQuotes/Terminal/Common/Files:/data/mt4:ro"`
  then use `/data/mt4/history.htm`.
