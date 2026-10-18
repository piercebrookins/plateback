# Dining-hall demonstration

The app opens on a campus dining overview, with the original laptop scanner retained under **Scan a plate** and real scan statistics under **Your scans**.

The demo contains **240 fictional plate-return records across six lunch sessions**. Four sessions serve a chicken/rice/broccoli meal; two serve pasta/broccoli. There are 40 simulated sampled returns and 200 simulated meals served per session. These are deterministic demonstration fixtures, not observations from UVA or another institution. No student data is used.

In the default scenario, rice remains on **116 of 160 rice-meal returns (72.5%)**. The UI rounds to 73%. Chicken remains on 42/160; broccoli on 54/240; pasta on 30/80. Food-specific denominators include only meals serving that food. With East dining selected, rice is 89/120 (74.2%). All 240 leftover weights are null/unknown. “Substantial rice” is a synthetic reviewer category, not a quantity inferred from photos.

The photo gallery uses Nutrition5k official-training images matched to the meal family (rice/chicken or pasta). These are served-food photographs, not student return images; photos repeat and each is visibly marked illustrative. Attribution: Thames et al., CVPR 2021, CC BY 4.0, via the mmathys/food-nutrients mirror. The sample rates were not calculated from those photos or model predictions. The independent model benchmark uses the official test split instead.

Hall and meal filters update the metrics, comparison table and gallery together. Food bars filter the plate records; **Load 24 more plates** pages through the full gallery. Selecting a plate shows its simulated observation, unknown weight, photo source and provenance. CSV export includes only the currently filtered gallery records and identifies them as fictional.

**Draft a portion trial** saves a local proposal for a 10%, 15% or 20% smaller default scoop with free seconds. It proposes comparable menu/service windows, a rice-leftover presence measure, and satisfaction/second-helping guardrails. It does not schedule operational changes or generate savings or causal claims. Repeated leftovers motivate a test; preferences, taste and timing remain alternative explanations. Trial proposal state persists in this browser's local storage, separate from the real scan database.

Fixture generation: `.venv/bin/python scripts/seed_dining_demo.py`. Sources: `data/demo-dining.json` and `ml/nutrition-manifest.json`. This fixture is isolated from `/api/dashboard`, real scan history and all benchmark ground truth.
