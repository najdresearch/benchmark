# Arabic support routing — development fixture

Eight original synthetic messages exercise the dataset → benchmark → Arena contracts.
They test routing to billing, delivery, returns or a human agent. They do not measure
support resolution, Saudi knowledge, policy compliance, or production model quality.

These original examples are dedicated under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).
They contain no customer, employer or private evaluation material. The labels are authored
with the templates, not independently reviewed. The split is public development, never
held-out evaluation. The always-billing baseline is 2/8 (25%). Eight examples are too few
for model rankings; use this pack only to verify integration.

Rebuild with `uv run python scripts/generate_support_contract_example.py`.
The JSON manifest pins case bytes, IDs, count, origin and rights. The task pack pins the
prompt, label contract, scorer, one-attempt budget and stop rules. Changing cases requires
a new dataset version; changing the prompt or scoring contract requires a new task version.
