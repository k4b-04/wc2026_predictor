# World Cup 2026 Round of 32
"Actual" score shown is the **90-minute regulation scoreline**

## Summary

| Metric | Result |
| --- | --- |
| Outcome accuracy | **13 / 16 correct (81.3%)** |
| Exact scoreline accuracy | **6 / 16 correct (37.5%)** |
| Draw calls correct | **3 / 4 (75%)** |

Both figures are well above this model's historical benchmarks (~60% outcome accuracy, ~10-15% exact score in prior batch tests).

## Match-by-match results

| Match | Predicted | Predicted Score | Actual Score | Outcome | Score |
| --- | --- | --- | --- | --- | --- |
| South Africa vs Canada | Canada Win | 1–2 | 0–1 | ✅  | ❌ |
| Brazil vs Japan | Brazil Win | 2–1 | 2–1 | ✅  | ✅ Exact |
| Germany vs Paraguay | Germany Win | 2–1 | 1–1* | ❌  | ❌ |
| Netherlands vs Morocco | Draw | 1–1 | 1–1* | ✅  | ✅ Exact |
| Ivory Coast vs Norway | Norway Win | 1–2 | 1–2 | ✅  | ✅ Exact |
| France vs Sweden | France Win | 2–1 | 3–0 | ✅  | ❌ |
| Mexico vs Ecuador | Draw | 1–1 | 2–0 | ❌  | ❌ |
| England vs DR Congo | England Win | 2–1 | 2–1 | ✅  | ✅ Exact |
| Belgium vs Senegal | Draw | 1–1 | 2–2* | ✅  | ❌ |
| United States vs Bosnia and Herzegovina | United States Win | 2–1 | 2–0 | ✅  | ❌ |
| Spain vs Austria | Spain Win | 2–1 | 3–0 | ✅  | ❌ |
| Portugal vs Croatia | Portugal Win | 2–1 | 2–1 | ✅  | ✅ Exact |
| Switzerland vs Algeria | Switzerland Win | 2–1 | 2–0 | ✅  | ❌ |
| Australia vs Egypt | Draw | 1–1 | 1–1* | ✅  | ✅ Exact |
| Argentina vs Cape Verde | Argentina Win | 3–0 | 1–1* | ❌  | ❌ |
| Colombia vs Ghana | Colombia Win | 2–0 | 1–0 | ✅  | ❌ |

\* Went to extra time and/or penalties

## Notable misses

- **Germany vs Paraguay**: predicted Germany to win 2–1. Germany were held 1–1 and eliminated on penalties. This was likely due to Paraguay's defensive gameplan (25% possession, disciplined low block), which isn't captured by the season-form and ranking features.
- **Mexico vs Ecuador**: predicted a tight 1–1 draw. Mexico won comfortably 2–0. This was likely due to the co-host home advantage at the Azteca.
- **Argentina vs Cape Verde**: predicted a routine 3–0 Argentina win. Cape Verde forced extra time and the match was level 1–1 after 90 minutes. This result was seen as underwhelming from Argentina's perspective, given FIFA ranks Argentina at #1 and Cape Verde at #67.

## Notable hits

- Six exact scorelines correctly called: **Brazil 2–1 Japan**, **Netherlands 1–1 Morocco**, **Ivory Coast 1–2 Norway**, **England 2–1 DR Congo**, plus **Portugal 2–1 Croatia** and **Australia 1–1 Egypt** — six exact hits in total.
- 3 of the 4 draw calls were correct (Netherlands–Morocco, Belgium–Senegal, Australia–Egypt). All three correct draw calls were also matches that went to penalties, suggesting the model's draw signal may be more informative for evenly-matched knockout ties than the aggregate walk-forward numbers implied.
