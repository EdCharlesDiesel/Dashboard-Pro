import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { HttpClientModule } from '@angular/common/http';
import { ReactiveFormsModule } from '@angular/forms';
import { CommonModule, DecimalPipe, DatePipe } from '@angular/common';

// ── Angular Material ──────────────────────────────────────────────────────────
import { MatButtonModule }      from '@angular/material/button';
import { MatCardModule }        from '@angular/material/card';
import { MatChipsModule }       from '@angular/material/chips';
import { MatDialogModule }      from '@angular/material/dialog';
import { MatDividerModule }     from '@angular/material/divider';
import { MatFormFieldModule }   from '@angular/material/form-field';
import { MatIconModule }        from '@angular/material/icon';
import { MatInputModule }       from '@angular/material/input';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { MatSnackBarModule }    from '@angular/material/snack-bar';
import { MatTableModule }       from '@angular/material/table';
import { MatTooltipModule }     from '@angular/material/tooltip';

// ── App ───────────────────────────────────────────────────────────────────────
import { AppComponent }                  from './app.component';
import { TradePlanDashboardComponent }   from './trade-plan-dashboard/trade-plan-dashboard.component';
import { ClosePlanDialogComponent }      from './trade-plan-dashboard/close-plan-dialog.component';
import { CountDirectionPipe }            from './trade-plan-dashboard/count-direction.pipe';

@NgModule({
  declarations: [
    AppComponent,
    TradePlanDashboardComponent,
    ClosePlanDialogComponent,
    CountDirectionPipe,
  ],
  imports: [
    BrowserModule,
    BrowserAnimationsModule,
    HttpClientModule,
    ReactiveFormsModule,
    CommonModule,

    // Material
    MatButtonModule,
    MatCardModule,
    MatChipsModule,
    MatDialogModule,
    MatDividerModule,
    MatFormFieldModule,
    MatIconModule,
    MatInputModule,
    MatProgressBarModule,
    MatSnackBarModule,
    MatTableModule,
    MatTooltipModule,
  ],
  providers: [DecimalPipe, DatePipe],
  bootstrap: [AppComponent],
})
export class AppModule {}
