# Alpha Lines v6.2: pipeline fixes from the Codex review (2026-10-13)

Status: **staged, not live.** The weekly pipeline still runs release `v5.5-2026-10-03`. Nothing in Drive, the Code Vault or the scheduled task has changed.

Scope: pipeline hygiene only. No prediction, price, edge rule or gate rule changed. The CLV gate statistics and a quote-age limit are deliberately left out, pending a decision.

## What changed, by fix

| # | Fix | Files | Behavior now |
|---|---|---|---|
| 1 | Incomplete moneyline | checker.py, pricer.py | Fixtures the Pricer flags `missing_h2h` are left out of the price checks and listed as **WARN `market.skipped`**. They BLOCK only if a paper pick is on one. Before, they were summed as 0 and blocked the whole run ("off by 1.0000", reproduced on the old code). Missing or non-numeric probabilities, overround or claimed implied goals on any other row now BLOCK (NaN used to slip through comparisons). |
| 2 | Baseline can't forget a season | checker.py, publish.py | `hist.baseline` BLOCKs when a season file in the baseline has disappeared. `publish.py --confirm` refuses to write a baseline that drops one. To retire a season on purpose, edit the baseline by hand. |
| 3 | Checker reads upstream results | checker.py, manifest.py | New `upstream.collector/pricer/runner/banker` checks (with `--run-dir`). Each needs a manifest written for this run, unchanged outputs, and no unlisted files (else BLOCK). FAILED → BLOCK, PARTIAL → WARN carrying its warnings. Inputs a run requires no longer SKIP: `priced.csv` or `paper_picks.csv` absent → BLOCK, `ratings.json` absent → WARN, empty `priced.csv` outside a break → WARN. Repo mode (no `--run-dir`) keeps the old SKIP behavior. |
| 4 | Publisher stricter | publish.py, manifest.py | `prepare` requires all five upstream manifests (collector, pricer, runner, banker, checker), each written by the right agent for this run, none FAILED, outputs unchanged, no unlisted files. `confirm` re-verifies the collector manifest before building the baseline from it. New helpers: `manifest.check_manifest()` and `verify_outputs(strict=True)`. |
| 5 | Run IDs | manifest.py, orchestrate.py, collect.py | IDs are UTC time to the second plus a 4-hex suffix, e.g. `2026-10-14T042736Z-9799`. The orchestrator refuses an existing run folder (exit 1, "ORCHESTRATOR REFUSED"). The collector refuses a non-empty collector folder. |
| 6 | Banker validates inputs | banker.py | New `screen()` runs before sizing. It rejects a price that isn't finite and > 1, a probability outside (0, 1), an edge that is non-positive, or an edge below the runner's `edge_threshold` (0.02). Rejects go to `not_sized` with the reason, plus a manifest warning. A Kelly score of 0 gets 0 units (it used to get the 1-unit minimum) and doesn't stop the cap walk. |

## Round 2 (after Codex's review of the first v6.2 diff)
| Codex item | Done |
|---|---|
| 2. Nested files escape strict checks | `verify_outputs(strict=True)` now reports any sub-folder or symlink in an agent folder. The Publisher refuses top-level entries in the run folder other than the agent folders, `publisher`, `orchestrator` and `bankroll`. |
| 3. Goals schema still SKIPs in run mode | In run mode, `lam`, `mu` and `goals_basis` must exist (else BLOCK). `goals_basis` must be `none` or `1x2+total`. `none` with implied goals filled in → BLOCK. `1x2+total` with them missing or out of range → BLOCK. |
| 4. Banker without a valid edge threshold | A missing, non-finite or non-positive `edge_threshold` → Banker FAILED when the gate is open. When the gate is closed it's a warning, and every candidate is rejected (no preview), so the Banker can't fall back to "any positive edge". |
| 1. Which warnings block | **Pending Alcadio's decision** (proposal sent). |

## Cleanups
- **Atomic baseline write:** `manifest.atomic_write_text` (temp file, fsync, `os.replace`).
- **Nested blend weights:** the Checker now inspects every numeric leaf, not just top-level values.
- **Units type:** `bank.tiers` rejects booleans and non-integers, so `True` no longer passes as 1.
- **Dead check removed:** the `home_attempts` check (the Pricer never wrote that column).
- **Exact flags:** the Pricer counts flags by exact name, not substring.
- **Verdict required:** the orchestrator lets the Publisher run only if a PASS/WARN `verdict.json` for this run folder exists, not just exit code 0.
- **Strict gate parsing:** `runner._runtime_gate` accepts exactly one `GATE_OPEN = True|False` line, and anything else fails the run.
- **Filename validation:** history files must be named `CODE_SSSS.csv`. The collector reports others as a problem (no crash), and the Checker BLOCKs `hist.names`.

## Not done (left on purpose)
- **CLV gate statistics:** clustering by match or week, and same-selection closes only. Needs Alcadio's decision.
- **Quote-age limit for snapshots:** needs a threshold.
- **Brittle text matching in collect.py:** FAILED vs PARTIAL is still decided by matching error text. It works today, so I didn't restructure it.
- **`sys.path` / module-global mutation in runner.py:** low risk, since agents run as separate processes.
- **Ledger status validation in banker.py:** the dashboard's full set of bet statuses isn't confirmed, so a strict check could reject real records.

## Tests
- **New:** `test_v62.py`, 48 tests after round 2 (47 + 1 end-to-end that runs only when `PULSEMATCH_REPO` points at a checkout). All pass, including the end-to-end run on `alcadiomontes03/match-engine-odds` @ 330df82. Against the old v6 code, 34 of them fail, so they detect each bug; the other 6 are regression guards.
- **Real data:** a full orchestrator run on the public repo gives verdict WARN, 0 blocking, publish bundle ready, the same outcome as the old code. Collector and Pricer warnings now appear in the verdict.
- **Original 75 tests:** **not yet re-run.** They live only on Drive. Some will need updating where behavior intentionally changed:
  - a Publisher test that expects to publish with only a runner manifest;
  - a Checker `--run-dir` test without upstream manifests;
  - the run-ID format;
  - a Banker test expecting 1 unit for a zero-edge pick.

## Release steps still to do (README release process)
1. Re-run and update the original 6 test suites.
2. Upload the 8 agents and the tests to Drive `agents/`. Move the old files to `superseded/`.
3. Create `releases/v6.2-2026-10-13/` with the copies and `RELEASE.json`.
4. Re-seed the Code Vault (`agents` collection) from files, then read it back and hash-check.
5. Update the weekly task's table of ids, bytes and sha256, and check whether its prompt passes a `--run-id` in the old format.
6. Mirror a note into the Project docs.
