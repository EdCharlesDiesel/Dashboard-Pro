import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface TradePlan {
  id: string;
  pair: string;
  direction: 'Long' | 'Short' | 'None';
  status: 'Pending' | 'Active' | 'Closed' | 'Cancelled' | 'StoppedOut';
  timeframe: string;
  entryPrice: number;
  stopLoss: number;
  takeProfit1: number;
  takeProfit2: number;
  closePrice?: number;
  pnL?: number;
  riskReward: number;
  atr: number;
  ema20: number;
  ema50: number;
  rsi: number;
  support: number;
  resistance: number;
  reasoning: string;
  openedAt: string;
  closedAt?: string;
}

export interface GenerateResponse {
  generated: number;
  plans: TradePlan[];
}

export interface ClosePlanRequest {
  closePrice: number;
}

@Injectable({ providedIn: 'root' })
export class TradePlanService {
  private readonly base = 'https://localhost:7001/api/trade-plans';

  constructor(private http: HttpClient) {}

  generate(): Observable<GenerateResponse> {
    return this.http.post<GenerateResponse>(`${this.base}/generate`, {});
  }

  getPending(): Observable<TradePlan[]> {
    return this.http.get<TradePlan[]>(`${this.base}/pending`);
  }

  getById(id: string): Observable<TradePlan> {
    return this.http.get<TradePlan>(`${this.base}/${id}`);
  }

  close(id: string, closePrice: number): Observable<TradePlan> {
    return this.http.patch<TradePlan>(`${this.base}/${id}/close`, { closePrice });
  }
}
