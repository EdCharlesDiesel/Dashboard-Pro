// ── count-direction.pipe.ts ───────────────────────────────────────────────────
// ng generate pipe count-direction
import { Pipe, PipeTransform } from '@angular/core';
import { TradePlan } from './trade-plan.service';

@Pipe({ name: 'countDirection' })
export class CountDirectionPipe implements PipeTransform {
  transform(plans: TradePlan[], direction: string): number {
    return plans.filter(p => p.direction === direction).length;
  }
}
