import { Component, OnInit, OnDestroy } from '@angular/core';
import { MatDialog } from '@angular/material/dialog';
import { MatSnackBar } from '@angular/material/snack-bar';
import { Subject, interval } from 'rxjs';
import { takeUntil, switchMap, startWith } from 'rxjs/operators';
import { TradePlan, TradePlanService } from './trade-plan.service';
import { ClosePlanDialogComponent } from './close-plan-dialog.component';

@Component({
  selector: 'app-trade-plan-dashboard',
  templateUrl: './trade-plan-dashboard.component.html',
  styleUrls:  ['./trade-plan-dashboard.component.scss']
})
export class TradePlanDashboardComponent implements OnInit, OnDestroy {
  plans: TradePlan[]  = [];
  selected: TradePlan | null = null;
  generating = false;
  loading    = false;

  readonly displayedColumns = [
    'pair', 'direction', 'timeframe',
    'entryPrice', 'stopLoss', 'takeProfit1', 'takeProfit2',
    'riskReward', 'rsi', 'status', 'actions'
  ];

  private destroy$ = new Subject<void>();

  constructor(
    private svc:     TradePlanService,
    private dialog:  MatDialog,
    private snack:   MatSnackBar
  ) {}

  ngOnInit(): void {
    // Poll pending plans every 30 seconds
    interval(30_000)
      .pipe(startWith(0), takeUntil(this.destroy$))
      .subscribe(() => this.loadPlans());
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  loadPlans(): void {
    this.loading = true;
    this.svc.getPending().subscribe({
      next: plans => {
        this.plans   = plans;
        this.loading = false;
        // Refresh selected if it's still in the list
        if (this.selected) {
          this.selected = plans.find(p => p.id === this.selected!.id) ?? null;
        }
      },
      error: () => {
        this.loading = false;
        this.snack.open('Failed to load plans', 'Dismiss', { duration: 3000 });
      }
    });
  }

  generate(): void {
    this.generating = true;
    this.svc.generate().subscribe({
      next: res => {
        this.generating = false;
        this.snack.open(
          res.generated > 0
            ? `✓ ${res.generated} plan(s) generated`
            : 'No new setups found — market conditions not met',
          'OK',
          { duration: 4000 }
        );
        this.loadPlans();
      },
      error: () => {
        this.generating = false;
        this.snack.open('Generation failed', 'Dismiss', { duration: 3000 });
      }
    });
  }

  select(plan: TradePlan): void {
    this.selected = this.selected?.id === plan.id ? null : plan;
  }

  openClose(plan: TradePlan, event: Event): void {
    event.stopPropagation();
    const ref = this.dialog.open(ClosePlanDialogComponent, {
      data: plan,
      width: '420px'
    });

    ref.afterClosed().subscribe((closePrice?: number) => {
      if (!closePrice) return;
      this.svc.close(plan.id, closePrice).subscribe({
        next: () => {
          this.snack.open('Position closed', 'OK', { duration: 3000 });
          this.selected = null;
          this.loadPlans();
        },
        error: () => this.snack.open('Close failed', 'Dismiss', { duration: 3000 })
      });
    });
  }

  // ── Helpers ────────────────────────────────────────────────────────────────

  directionColor(d: string): string {
    return d === 'Long' ? 'accent' : d === 'Short' ? 'warn' : 'primary';
  }

  statusColor(s: string): string {
    const map: Record<string, string> = {
      Pending: '#f59e0b', Active: '#3b82f6',
      Closed:  '#6b7280', Cancelled: '#6b7280', StoppedOut: '#ef4444'
    };
    return map[s] ?? '#6b7280';
  }

  rsiLabel(rsi: number): string {
    if (rsi >= 70) return 'Overbought';
    if (rsi <= 30) return 'Oversold';
    if (rsi >= 55) return 'Bullish';
    if (rsi <= 45) return 'Bearish';
    return 'Neutral';
  }

  rsiColor(rsi: number): string {
    if (rsi >= 70 || rsi <= 30) return '#ef4444';
    if (rsi >= 55) return '#22c55e';
    if (rsi <= 45) return '#f97316';
    return '#3b82f6';
  }

  rrClass(rr: number): string {
    if (rr >= 2.5) return 'rr-excellent';
    if (rr >= 1.5) return 'rr-good';
    return 'rr-weak';
  }

  trendSignal(plan: TradePlan): string {
    if (plan.ema20 > plan.ema50) return '↑ Uptrend';
    if (plan.ema20 < plan.ema50) return '↓ Downtrend';
    return '→ Sideways';
  }

  trendColor(plan: TradePlan): string {
    if (plan.ema20 > plan.ema50) return '#22c55e';
    if (plan.ema20 < plan.ema50) return '#ef4444';
    return '#6b7280';
  }

  supportDistancePct(plan: TradePlan): number {
    return Math.abs((plan.entryPrice - plan.support) / plan.entryPrice) * 100;
  }

  resistanceDistancePct(plan: TradePlan): number {
    return Math.abs((plan.resistance - plan.entryPrice) / plan.entryPrice) * 100;
  }
}
