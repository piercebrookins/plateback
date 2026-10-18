# OpenAI label review: GPT-6 Luna

Configured model: `gpt-6-luna`, using the Responses API with `reasoning.effort: none`, Standard (`default`) service tier, JSON output, 2,000 output-token maximum, and no automatic retries. The local segmentation model creates the geometry; Luna reviews the whole image and masked crops and returns editable label suggestions. It never supplies measured weight.

The user selected Luna. Its low token price suits this bounded label-review task. This is a cost/task-fit choice, not a measured claim that Luna is more accurate than the previous provider model. No head-to-head label-review accuracy benchmark was performed. The local model's evaluation is independent of cloud suggestions.

Official references checked October 17, 2026: [model documentation](https://developers.openai.com/api/docs/models/gpt-6-luna), [pricing](https://developers.openai.com/api/docs/pricing). Standard short-context rates per million tokens: $0.10 ordinary input, $0.01 cached input, $0.125 cache writes, and $0.50 output. Requests are bounded far below the long-context threshold. The budget calculator accounts for cached reads and writes separately and retains historical charges unchanged.

The real integration call on independent reference crops from validation image `train_1031` completed: 562 input tokens, 68 output tokens, no cache or reasoning tokens, estimated cost **$0.0000902**. It returned rice and chicken suggestions. This verifies request/response integration, not classifier accuracy or species identification. Evidence: `artifacts/openai-smoke-gpt-6-luna.json`. Existing GPT-4.1-mini scan history retains its original model attribution.

The `.env` key remains server-only. Gemini remains disabled until its key and priced model are supplied. Paid GPU spending is zero; local training makes no API calls.
