import { Component, Inject } from '@angular/core';
import { FormControl, Validators } from '@angular/forms';
import { MAT_DIALOG_DATA, MatDialogRef } from '@angular/material/dialog';
import { TradePlan } from './trade-plan.service';

@Component({
  selector: 'app-close-plan-dialog',
  template: `
    <h2 mat-dialog-title class="dialog-title">
      Close {{ data.pair }} {{ data.direction }}
    </h2>

    <mat-dialog-content class="dialog-content">
      <div class="plan-summary">
        <div class="summary-row">
          <span class="label">Entry</span>
          <span class="value">{{ data.entryPrice | number:'1.3-3' }}</span>
        </div>
        <div class="summary-row">
          <span class="label">Stop Loss</span>
          <span class="value danger">{{ data.stopLoss | number:'1.3-3' }}</span>
        </div>
        <div class="summary-row">
          <span class="label">TP1</span>
          <span class="value success">{{ data.takeProfit1 | number:'1.3-3' }}</span>
        </div>
      </div>

      <mat-form-field appearance="outline" class="full-width">
        <mat-label>Close Price</mat-label>
        <input matInput type="number" [formControl]="closePrice"
               placeholder="{{ data.entryPrice }}">
        <mat-error *ngIf="closePrice.hasError('required')">Required</mat-error>
        <mat-error *ngIf="closePrice.hasError('min')">Must be greater than 0</mat-error>
      </mat-form-field>

      <div class="pnl-preview" *ngIf="closePrice.valid && closePrice.value">
        <span class="label">Estimated P&L</span>
        <span class="value" [class.success]="estimatedPnl > 0" [class.danger]="estimatedPnl < 0">
          {{ estimatedPnl > 0 ? '+' : '' }}{{ estimatedPnl | number:'1.3-3' }}
        </span>
      </div>
    </mat-dialog-content>

    <mat-dialog-actions align="end">
      <button mat-button mat-dialog-close>Cancel</button>
      <button mat-flat-button color="primary"
              [disabled]="closePrice.invalid"
              (click)="confirm()">
        Close Position
      </button>
    </mat-dialog-actions>
  `,
  styles: [`
    .dialog-title { font-weight: 600; }
    .dialog-content { min-width: 320px; }
    .plan-summary {
      background: rgba(0,0,0,0.04);
      border-radius: 8px;
      padding: 12px 16px;
      margin-bottom: 20px;
    }
    .summary-row {
      display: flex;
      justify-content: space-between;
      padding: 4px 0;
    }
    .label { color: rgba(0,0,0,0.54); font-size: 13px; }
    .value { font-weight: 600; font-size: 13px; }
    .success { color: #2e7d32; }
    .danger  { color: #c62828; }
    .full-width { width: 100%; margin-top: 8px; }
    .pnl-preview {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 8px 0;
      border-top: 1px solid rgba(0,0,0,0.12);
      margin-top: 4px;
    }
  `]
})
export class ClosePlanDialogComponent {
  closePrice = new FormControl<number | null>(null, [
    Validators.required,
    Validators.min(0.001)
  ]);

  constructor(
    public dialogRef: MatDialogRef<ClosePlanDialogComponent>,
    @Inject(MAT_DIALOG_DATA) public data: TradePlan
  ) {}

  get estimatedPnl(): number {
    const cp = this.closePrice.value;
    if (!cp) return 0;
    return this.data.direction === 'Long'
      ? cp - this.data.entryPrice
      : this.data.entryPrice - cp;
  }

  confirm(): void {
    if (this.closePrice.valid && this.closePrice.value) {
      this.dialogRef.close(this.closePrice.value);
    }
  }
}
