# Two-minute demo

Use localhost and label any public dataset photograph **demo**. Never call public food photography a live cafeteria observation.

- **0:00–0:15**: “PlateBack helps a dining team see which foods recur in sampled leftovers. Today we have a laptop, so weight stays unknown.”
- **0:15–0:45**: Upload a public mixed-food test image, mark it demo, choose Local, analyze. Explain that these are masks from a locally fine-tuned model, not boxes. A suitable successful held-out example is `data/datasets/plateback/images/test/test_49.jpg`; disclose that it is a curated demo example, not a random accuracy estimate.
- **0:45–1:15**: Select and rename a mask, move a vertex, delete a false detection or draw a missed region. Toggle Original to show predictions are preserved. Confirm; reload to show persistence. Leave grams blank.
- **1:15–1:35**: Open Dashboard. Real scans remain zero unless real plates were captured. Enable demo scans deliberately to show category-presence bars. Point out the sample count and “Weight not measured.”
- **1:35–1:50**: Show the saved proposal: smaller rice portions with free seconds. Explain that comparable sessions and sampling are needed to assess it.
- **1:50–2:10**: Show the evaluation report: 88 held-out public images, 52.9% macro mask IoU, 12/20 other-food false-positive cases. Chicken/duck is a combined source label. This is a reviewed prototype, not automated waste weighing.

Optional: show the previously saved Local + OpenAI result. The OpenAI call has been exercised with real credentials; Gemini controls remain disabled until a key is provided. Do not imply an untested Gemini demo occurred.

No video recording or public repository publication has been performed. The code, screenshots, evaluation artifacts and this walkthrough support recording your own submission.
