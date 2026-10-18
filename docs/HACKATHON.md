# PlateBack at UVA

Run `./scripts/start.sh` and open http://127.0.0.1:8000/.

1. In Settings, choose OpenAI or PlateBack. OpenAI with Astra is the initial browser default. Luna is available in Settings at lower cost. See the [small scanner comparison](ASTRA_SCANNER_CHECK.md). PlateBack uses the semantic-v4 checkpoint; version labels stay out of the interface.
2. Open Scan a plate. Choose O-Hill, Newcomb, Runk, or a meal session, then upload a photo or use the camera.
3. Click Analyze plate. Cloud analysis sends the photo to OpenAI. Local analysis stays on the laptop.
4. Check every food outline. Relabel, move points, delete incorrect regions, or draw missed food. OpenAI also guesses other food names. Pieces with the same name share one row; confirm, rename or remove each guess. PlateBack local mode still recognizes four target foods.
5. Confirm plate, then open Your scans or History. Weight controls are hidden; new scans keep weight unknown.

The overview leads with food remaining on returned plates, followed by the corresponding records. Hall names follow [UVA's welcome schedule](https://orientation.virginia.edu/2026-wahoo-welcome-schedule). Settings controls the fictional UVA scenario and 1,200 sample records. They are not actual UVA observations; photos are illustrative and may repeat. Samples never enter training or evaluation.

PlateBack's local model has 54.12% mask IoU on the 12-image pilot, but only 30% chicken detection recall on the original 38-image Nutrition5k presence set. The model can miss chicken. The original encoder has no food segmentation head, so its pretrained food score is unavailable.

A successful upload and save does not establish segmentation accuracy. Scanner outputs need visual review. The global API cap remains $5; failed or uncertain requests retain their reservations. Physical camera permission and image quality depend on the laptop and lighting.
