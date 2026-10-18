# PlateBack

PlateBack tracks food left on plates. Upload a photo or capture one with a laptop camera, review suggested food labels and outlines, and save the plate. The overview groups recurring leftovers across UVA dining-hall sample views.

Built with React, TypeScript, Vite, and FastAPI. Analysis supports OpenAI and a locally trained segmentation model.

## Run locally

Install Node.js 20.19+ or 22.12+ and [uv](https://docs.astral.sh/uv/). Setup creates a Python 3.11 environment, installs pinned dependencies, builds the frontend, and prepares bundled demo records.

```sh
cp .env.example .env
./scripts/setup.sh
./scripts/start.sh
```

Open http://127.0.0.1:8000. Copy the environment template only on initial setup; preserve an existing .env when updating.

The overview, sample gallery, and benchmark work with bundled fixtures. Automatic scanning needs an OpenAI API key in .env or local weights. Keys stay on the server. Select the provider in Settings. 

For frontend development, start the backend, then run `npm run dev` in frontend/. This prototype binds to localhost and has no authentication; it is not configured for public hosting.

## Local model

The selected model is semantic-v4, displayed as PlateBack. Checkpoints and training datasets are excluded from this repository. To use an existing checkpoint, place best.pt in artifacts/models/semantic-v4/. Setup installs its settings.json, and .env.example selects that path.

To reproduce training, see [the v4 protocol](docs/EVALUATION_V4_PROTOCOL.md), [evaluation](docs/EVALUATION_V4.md), and ml/. Research scripts can download datasets, train models, or call paid providers; they are separate from normal setup.

## Data and benchmark

Sample records are fictional. Their public Nutrition5k photos are illustrative served-food images, not measured UVA returns. Actual scans are stored separately in a local SQLite database. Disable sample data in Settings to view only your scans.

PlateBack scored 54.1% four-food mean mask IoU on twelve FoodSeg103 images. Chicken labels include duck. This small pilot does not establish general model superiority or performance on dining-hall leftovers. See [the protocol](docs/BENCHMARK_PROTOCOL.md) and [asset attribution](assets/README.md).

Food presence and image area are not food weight. Unmeasured weights remain unknown; the app does not infer savings or waste reductions from images.

## Tests

```sh
.venv/bin/python -m pytest -q
cd frontend
npm run build
```

## Project layout

- backend/: API, analysis providers, storage, and budget controls.
- frontend/: React application.
- assets/: sample fixtures, photos, and recorded benchmark results.
- ml/: training, dataset manifests, and evaluation scripts.
- scripts/: setup, startup, and research utilities.
- tests/: backend and model tests.
- docs/: design, provenance, and experiment reports. Historical reports describe their own versions.

Uploads, databases, checkpoints, datasets, dependencies, and generated output are ignored by Git.

## License

Application code is AGPL-3.0; see [LICENSE](LICENSE). Third-party data and models have their own terms. See [data and licenses](docs/DATA_AND_LICENSES.md) and [photo attribution](assets/README.md).
