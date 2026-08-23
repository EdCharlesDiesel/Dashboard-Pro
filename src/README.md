# Trade Plan Dashboard — Angular Material

## Setup

```bash
# 1. Create new Angular app (skip if you already have one)
ng new orion-dashboard --style=scss --routing=false
cd orion-dashboard

# 2. Add Angular Material
ng add @angular/material
# Choose: Indigo/Pink theme, Yes to typography, Yes to animations

# 3. Copy these files into src/app/trade-plan-dashboard/
#    - trade-plan.service.ts
#    - trade-plan-dashboard.component.ts
#    - trade-plan-dashboard.component.html
#    - trade-plan-dashboard.component.scss
#    - close-plan-dialog.component.ts
#    - count-direction.pipe.ts

# 4. Replace src/app/app.module.ts with the provided app.module.ts

# 5. Update src/app/app.component.html
echo '<app-trade-plan-dashboard></app-trade-plan-dashboard>' > src/app/app.component.html

# 6. Add Material Icons to index.html
# In <head>:
# <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
# <link href="https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&display=swap" rel="stylesheet">

# 7. Run
ng serve
```

## API Configuration

Update the base URL in `trade-plan.service.ts` to match your API:

```typescript
private readonly base = 'https://localhost:7001/api/trade-plans';
```

Your API must have CORS enabled for `http://localhost:4200` — which it already does
from your Program.cs `AngularApp` policy.

## File Structure

```
src/app/
├── app.component.html          ← <app-trade-plan-dashboard>
├── app.module.ts               ← all Material imports
└── trade-plan-dashboard/
    ├── trade-plan.service.ts              ← HTTP calls to your API
    ├── trade-plan-dashboard.component.ts  ← logic
    ├── trade-plan-dashboard.component.html← template
    ├── trade-plan-dashboard.component.scss← styles
    ├── close-plan-dialog.component.ts     ← close position dialog
    └── count-direction.pipe.ts            ← Long/Short count pipe
```

## Features

- **Generate Plans** — calls POST /api/trade-plans/generate, analyses latest candles
- **Pending table** — shows all open setups with entry/SL/TP/R:R/RSI
- **Detail panel** — click any row to see full indicator breakdown
  - EMA20 vs EMA50 trend signal
  - RSI with overbought/oversold labels
  - ATR weekly volatility
  - Support & Resistance distances
  - Price levels visual (resistance → TP2 → TP1 → entry → SL → support)
  - Reasoning text from the engine
- **Close position** — opens dialog with P&L preview, calls PATCH endpoint
- **Auto-refresh** — polls every 30 seconds
