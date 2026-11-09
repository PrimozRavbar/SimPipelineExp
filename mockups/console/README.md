# Console mockup (fake data)

A clickable prototype of the game's player console. It is not connected to the simulation: every number is generated in the page.

Open `index.html` in any browser (download it, then double-click). No server or install needed.

Screens, modelled on the tools ML engineers use:

| Screen | Real-world equivalent |
|---|---|
| Missions | Team goals / tickets |
| Pipeline | Airflow, Dagster, Kubeflow |
| Models | MLflow, Weights & Biases |
| Monitoring | Grafana, Datadog |
| Incidents | PagerDuty |

Scenario: mission 2, "Bad deploy". Model v4 was auto-promoted because its offline Recall@20 beat v3, but the ANN index still holds v3 movie embeddings, so online CTR fell from about 30% to about 21%. The player can roll back, find the root cause in the incident, rebuild the index or turn on "Rebuild on every deploy", and promote v4 again. Press play in the top bar to run the clock.
