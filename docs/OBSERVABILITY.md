# Observability Setup (Live Logs + Metrics)

SmartDoc now supports OpenTelemetry export for logs, traces, and metrics.

## What You Get

- Request metrics:
  - `smartdoc.requests.total`
  - `smartdoc.requests.errors.total`
  - `smartdoc.requests.duration.seconds`
- Stage metrics:
  - `smartdoc.stage.duration.seconds`
  - `smartdoc.stage.alerts.total`
- Workflow trace span:
  - `smartdoc.workflow.run`
- Log export through OTLP (when enabled)

## Quick Start (Grafana Cloud Free)

1. Create a free Grafana Cloud stack.
2. In Grafana Cloud, copy:
   - OTLP endpoint (base URL)
   - OTLP auth credentials (instance ID + API token)
3. Set `.env`:

```dotenv
OTEL_ENABLED=true
OTEL_SERVICE_NAME=smartdoc-ai
OTEL_SERVICE_NAMESPACE=smartrag
OTEL_SERVICE_VERSION=dev

OTEL_TRACES_ENABLED=true
OTEL_METRICS_ENABLED=true
OTEL_LOGS_ENABLED=true
OTEL_METRICS_EXPORT_INTERVAL_MS=10000

OTEL_EXPORTER_OTLP_ENDPOINT=https://otlp-gateway-prod-us-central-0.grafana.net/otlp
OTEL_EXPORTER_OTLP_HEADERS=Authorization=Basic <base64(instance_id:api_token)>
```

4. Start SmartDoc:

```bash
python main.py
```

5. Generate traffic in the UI and open Grafana Explore/Dashboards.

## Cloud Free-Tier Notes (Verified February 16, 2026)

- GCP Cloud Operations:
  - Cloud Logging includes a free monthly ingestion allotment.
  - Cloud Monitoring includes free allotments depending on data type.
  - Docs: https://cloud.google.com/stackdriver/pricing
- AWS CloudWatch:
  - Free tier includes logs ingestion/storage and custom metric quotas.
  - Docs: https://aws.amazon.com/cloudwatch/pricing/
- Azure Monitor:
  - Free ingestion allotment plus retention/custom metric/API quotas.
  - Docs: https://azure.microsoft.com/en-us/pricing/details/monitor/

If you want strict cloud-native ingestion instead of Grafana Cloud, keep SmartDoc OTLP enabled and route via an OpenTelemetry Collector to your chosen cloud backend.

## Local Live Logs

PowerShell tail:

```powershell
Get-Content .\logs\app.log -Wait
```

## Troubleshooting

- `OpenTelemetry disabled (OTEL_ENABLED=false)`:
  - Set `OTEL_ENABLED=true`.
- Export errors/timeouts:
  - Verify endpoint and headers.
  - Increase `OTEL_EXPORT_TIMEOUT_S`.
- If you provide per-signal endpoints:
  - Use full paths ending in `/v1/traces`, `/v1/metrics`, `/v1/logs`.
