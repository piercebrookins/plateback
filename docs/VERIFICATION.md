V9: 38 tests and frontend build pass. All six v6-relative regression sets pass recall/FP gates; mask IoU improves slightly on all four mask sets. Native inference verifies four model-component hashes. API cost this run $0; total accounted $1.63087759.

V8 run: 35 tests pass, frontend build passes, desktop/mobile Models graph and metric selector checked. Native local inference uses v6 with three component hashes. No API calls; total accounted $1.63087759. See EVALUATION_V8.md.

# Verification record — 2026-10-17

## Automated checks

`python -m pytest -q`: **16 tests passed**. Coverage includes persistence, separate raw/corrected masks, idempotent upload and confirmation, demo exclusion, null versus measured zero, failed inference and explicit manual recovery, invalid geometry and weights, atomic concurrent budget reservations, asymmetric mask coordinates/holes, pinched-contour conversion, EXIF orientation, malformed uploads, conflicting idempotency keys, sessions/experiments and Gemini-disabled no-spend behavior.

`npm run build`: TypeScript checks and Vite production build passed. Python source compile checks passed. Dependencies are pinned.

## Real model and provider checks

- Local fine tuning ran on built-in MPS and saved loadable weights. 88-image held-out evaluation completed with no evaluator failures; independent publisher masks used as truth. Model metrics and limits are in `EVALUATION.md`.
- Fresh-process CPU adapter reloaded trained weights and segmented five mixed held-out images. First call including imports/model load: 1.322 s; later calls 17–23 ms. Geometry validation found a pixel-pinched contour; conversion now splits topology with `make_valid`, preserves holes, and retains raw bitmap PNGs in prediction metadata. A regression test covers this repair. No model or threshold was changed based on test examples.
- Real OpenAI reference-crop review passed: 720 input / 96 output tokens, estimated $0.0004416.
- Real HTTP **Local + OpenAI** hybrid passed on a validation mixed plate: six local regions, six validated label suggestions; local geometry retained; weight null. Total call latency 4.315 s including cold local inference and cloud review. Usage: 847 input / 299 output tokens, estimated $0.0008172.
- Total known API usage through this build: **$0.0012588 estimated**, below $5. No paid data calls or GPU rental. Ledger survives process restart; estimates use published token rates, not an account billing export.
- Gemini has no supplied key. The adapter's configuration, schema path, SDK construction and no-key/no-charge behavior are verified; live connectivity and accuracy remain **untested**. This intentionally honors the requested deferred Gemini setup.

## Browser walkthrough

Playwright CLI operated the built app in Chromium:

- Uploaded an actual public image, marked demo, drew a four-vertex polygon, renamed rice to broccoli, dragged its first vertex, confirmed and reloaded. The same coordinates and corrected label were restored; original predictions remained separate. On the real hybrid result, delete, undo and reset-to-prediction also worked before confirmation.
- Dashboard excluded that demo scan by default; enabling demo scans displayed it. Missing mass remained “Unknown / Weight not measured.”
- Created a clearly labeled demo meal session with coverage notes and saved an experiment proposal.
- Simulated a camera-permission rejection: error displayed and Upload remained available. Camera capture/upload also passed with a 640×480 synthetic canvas stream, and its media track stopped after capture. This is **not an assertion that the physical webcam was exercised**.
- Browser screenshots and snapshots are stored under `output/playwright/` and `.playwright-cli/` locally. The initial favicon 404 was fixed with an inline SVG favicon.

Real human correction time, physical webcam framing, actual food capture, physical scale accuracy, and deployment under concurrent stations are not measured. The server is a single local station bound to 127.0.0.1, with no public authentication. Test scans are explicitly demo data and not evidence of cafeteria sampling.

Re-running `ml/prepare.py` reproduced the saved 349-image manifest exactly. Download SHA-256 checks passed. A scan of distributable source/docs found none of the configured key values.

Temporary synthetic-camera and arbitrary-polygon test scans were archived to `artifacts/browser-fixtures.json` and removed from active history after verification. The genuine Local + OpenAI public-image result remains as a confirmed, explicitly labeled demo scan.

## Current revision: dining overview, v3 and model benchmark

- **21 backend tests pass.** Added tests for Luna request configuration/cache pricing, confidence/area filtering, demo isolation/denominators and OpenRouter sharing the same cap. The production TypeScript/Vite build passes.
- Browser checks verified the 240-record demo, East dining filter (89/120 rice returns), food filtering (116 rice records), gallery pagination from 24 to 48, plate provenance dialog, Escape/focus handling, and a locally persisted 20% portion-trial proposal. CSV export was read back: exactly 116 filtered rows, all explicitly fictional, all weight cells blank. Mobile pages have no horizontal document overflow at 390px; wide benchmark/meal tables scroll inside their containers.
- Demo images now use Nutrition5k official-training cafeteria photos matched to meal family. Image URLs include source IDs so regenerated fixtures cannot show stale cached photographs. Photos remain illustrative, not actual returns. Real scan analytics contain none of these 240 fixtures.
- Local v2 training completed 40 epochs on this Mac's MPS accelerator. A separate 939-image presence-head fit ran on local CPU for 66.5 seconds, selecting epoch 14 using validation loss. V3 uses validation-selected class-specific sources and a chicken presence gate. The final unused 83-image check and its missing-pasta limitation are documented in `EVALUATION_V3.md`.
- All four OpenRouter models completed the requested pilot calls. Invalid JSON/geometry remains accounted; strict and parser-normalized scores are both retained. Eight mandatory-reasoning setup failures retain $0.80 of conservative reservations. Generalists still lead the external pilot; no frontier-superiority claim is made. Results: `BENCHMARK_RESULTS.md`.
- A real HTTP v3 + GPT-6 Luna scan succeeded on validation image `train_1031`: six local contour regions, validated suggestions, null weight, 5.543 seconds including cold local loading and remote review. Evidence: `artifacts/hybrid-v3-smoke.json`. Gemini's direct adapter still awaits its own key; Gemini-through-OpenRouter was exercised only in the benchmark.
- Cumulative accounted API spending after that live check: **$1.25889789 / $5** (including the retained $0.80). Actual returned usage charges are lower; this is not an account billing export. No paid GPU or data service was used.
- Screenshots: `output/playwright/dining-overview.png`, `dining-plate-gallery.png`, `dining-mobile.png`, `model-benchmark.png`, and `model-benchmark-mobile.png`.

The initial camera verification above remains a synthetic-stream check, not physical webcam verification. No real dining-hall observations or real scale readings were invented. The original v1 report and historic provider attributions remain preserved.

Final browser regression: selected a v3 mask, deleted it, undid the deletion, and confirmed the demo scan. All six original regions were restored exactly and persisted; weight remained null. Real scan analytics still exclude both genuine public-image demo scans and all 240 synthetic overview records. Final source/docs/frontend key scan found no configured API key values.

## Local-model research follow-up: v4/v5

- Expanded to 1,732 FoodSeg103 training images, with 748 chicken/duck examples. Independent split audits passed, including a final chicken holdout checked against 2,662 earlier images plus within-test duplicates.
- Tested full DeepLab fine-tuning (local MPS; early-stopped after 14 epochs), DINOv2 dense-head transfer learning (local CPU; 40 epochs), inference resolution, recall-constrained calibration and connected boundary recovery.
- Added native-resolution serving/calibration equivalence checks, component-filter and boundary-growth tests, and a polygon-union regression test. **27 tests passed** with `.venv/bin/python -m pytest -q`.
- Fixed invalid-image upload handling after ML-library initialization; upload decoding retains the native Pillow loader instead of triggering optional plugin installation.
- Candidate native weights passed an isolated API upload → analyze → edit → restore → confirm flow. Original predictions were retained, demo records stayed out of real analytics, invalid uploads were rejected, and unmeasured grams stayed null. Evidence: `artifacts/semantic-v5-smoke.json`. No cloud calls were made by this test.
- New final comparison and earlier regression results are in `docs/EVALUATION_V5.md`. Candidate v5 remains experimental; active v3 is unchanged because some recall gates fail. Gemini retains better segmentation IoU.
- Total API accounting: **$1.62107589 / $5**, including old retained reservations. All new benchmark responses are cached. No paid GPU was used.

## Chicken recall follow-up (v6 research)

- 29 tests passed (`.venv/bin/python -m pytest -q`), including additive rescue preserving v5 pixels and classifier gating without inventing unsupported regions.
- Three CPU-only training methods completed. New data: 300 train, 60 validation, 60 test; independent hash/perceptual audit checked 420 new images against 2,694 earlier images and found no retained near duplicates.
- Frozen candidate evaluated on Nutrition new60/old38 and four segmentation regression sets (110, 32, 188, 83). Per-image results: `artifacts/evaluation-v6/detailed-results.json`.
- Actual candidate inference matched cached evaluation masks for all four classes on positive and negative Nutrition examples: `artifacts/evaluation-v6/serving-smoke.json`.
- Active v3 checkpoint remains byte-identical to its preserved baseline. No candidate promotion, UI changes, or paid requests. Total accounted budget remains $1.62107589 / $5, including retained uncertain reservations.
- Report: [EVALUATION_V6.md](EVALUATION_V6.md). Nutrition recall improved; Gemini mask IoU remains higher, and the strict cross-dataset gate still fails. Unmeasured food weights remain unknown.

## V7 local backbone training and hackathon OpenAI modes

- Six local MPS epochs fine-tuned 3,601,157 parameters (last two DINO transformer blocks, normalization, head). Validation chose epoch 5; averaged weights were also evaluated. No paid GPU or API training.
- 34 automated tests passed; production frontend build passed. The app health endpoint confirms local, research and OpenAI readiness. Default local weights remain v3.
- New modes: direct GPT-6 Luna contours (experimental) and research masks + OpenAI label review (experimental). Unknown weights remain null. Fragmented plates retain every local region; only the eight largest receive cloud review and the rest are explicitly marked for manual review.
- Two 24-image paid direct-OpenAI diagnostics retained all successes/failures. Normalized strict schema parsed 24/24; quality limits are documented, not inferred from parsing success.
- One real research + OpenAI call passed; isolated API replay verified editing/confirmation and demo isolation. `artifacts/research-openai-v7-smoke.json` records this separately from network tests.
- This run accounted $0.0098017 under its immutable $3 scope ceiling. Cumulative accounted total $1.63087759 / $5, including historical unknown-outcome reservations.
- Full regression results and failed deployment checks: [EVALUATION_V7.md](EVALUATION_V7.md). Nutrition chicken detection gains held; false positives did not improve and some mask IoU/pixel recall regressions remain. No default promotion.
