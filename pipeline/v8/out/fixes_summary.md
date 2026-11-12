# v8 fixes, walk-forward (2023-08 to 2026-11)

4,506 matches with a closing price; 28-day refits; lower log loss is better. Gap = model minus closing market (Shin). Change = variant minus the live v8 method, with a 95% interval from resampling whole 28-day folds.

| Variant | 1X2 log loss | Gap to market | Change vs live | 95% | Promoted-team matches: gap | Others: gap | Heavy favs: market / model | Over 2.5 | BTTS | Goals bias |
|---|---|---|---|---|---|---|---|---|---|---|
| v8_live | 0.9844 | +0.0221 | +0.0000 | +0.0000 to +0.0000 | +0.0267 (1211) | +0.0203 | 0.711 / 0.645 | 0.6695 | 0.6822 | -0.020 |
| v8_live+stretch | 0.9839 | +0.0216 | -0.0005 | -0.0014 to +0.0006 | +0.0259 (1211) | +0.0200 | 0.711 / 0.664 | 0.6694 | 0.6823 | -0.001 |
| nnls | 0.9843 | +0.0220 | -0.0000 | -0.0002 to +0.0001 | +0.0265 (1211) | +0.0204 | 0.711 / 0.643 | 0.6693 | 0.6820 | -0.020 |
| nnls+stretch | 0.9836 | +0.0213 | -0.0008 | -0.0018 to +0.0002 | +0.0253 (1211) | +0.0198 | 0.711 / 0.665 | 0.6692 | 0.6821 | +0.001 |
| nnls_prom5 | 0.9832 | +0.0209 | -0.0011 | -0.0024 to +0.0001 | +0.0224 (1211) | +0.0204 | 0.711 / 0.646 | 0.6688 | 0.6816 | -0.018 |
| nnls_prom5+stretch | 0.9825 | +0.0202 | -0.0019 | -0.0036 to -0.0002 | +0.0212 (1211) | +0.0198 | 0.711 / 0.669 | 0.6687 | 0.6817 | +0.005 |
| nnls_prom20 | 0.9836 | +0.0212 | -0.0008 | -0.0026 to +0.0010 | +0.0238 (1211) | +0.0203 | 0.711 / 0.646 | 0.6686 | 0.6815 | -0.016 |
| nnls_prom20+stretch | 0.9830 | +0.0207 | -0.0014 | -0.0035 to +0.0008 | +0.0231 (1211) | +0.0198 | 0.711 / 0.669 | 0.6685 | 0.6816 | +0.006 |

## By league, gap to market

| Variant | E0 | SP1 | D1 | F1 |
|---|---|---|---|---|
| v8_live | +0.0169 | +0.0230 | +0.0257 | +0.0236 |
| v8_live+stretch | +0.0168 | +0.0215 | +0.0255 | +0.0237 |
| nnls | +0.0168 | +0.0232 | +0.0256 | +0.0235 |
| nnls+stretch | +0.0166 | +0.0212 | +0.0251 | +0.0234 |
| nnls_prom5 | +0.0181 | +0.0216 | +0.0225 | +0.0219 |
| nnls_prom5+stretch | +0.0182 | +0.0194 | +0.0220 | +0.0217 |
| nnls_prom20 | +0.0202 | +0.0217 | +0.0219 | +0.0212 |
| nnls_prom20+stretch | +0.0210 | +0.0195 | +0.0214 | +0.0210 |

## Fitted stretch s (walk-forward)

| v                   |   min |   median |   max |
|:--------------------|------:|---------:|------:|
| nnls+stretch        |  0.96 |     1.12 |  1.19 |
| nnls_prom20+stretch |  0.96 |     1.13 |  1.18 |
| nnls_prom5+stretch  |  0.96 |     1.13 |  1.19 |
| v8_live+stretch     |  0.94 |     1.11 |  1.17 |

Live-method off-target shot coefficient across folds: min -0.0293, median -0.0203, share negative 100%.
