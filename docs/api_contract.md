# Illustrative identity-signal API contract

This is a mock contract for discussing data integration. It is not a Mastercard API specification and does not call an external service.

## Example request

```json
{
  "request_id": "pov-000123",
  "transaction": {
    "transaction_id": "tx-000123",
    "event_time": "2025-01-15T14:23:00Z"
  },
  "identity_context": {
    "email_token": "tokenized-value",
    "device_token": "tokenized-device"
  }
}
```

Use synthetic tokens in demos. Do not put real personal information or credentials in repository examples.

## Example response

```json
{
  "request_id": "pov-000123",
  "status": "partial",
  "signals": {
    "device_trust": 0.91,
    "email_risk": 0.12,
    "ip_proxy_risk": null
  },
  "signal_timestamp": "2025-01-15T14:23:00Z",
  "latency_ms": 84
}
```

Signal values are illustrative normalized scores from 0 to 1; their direction and semantics must be defined by the producing system. Null means unavailable, not zero risk.

## QA checklist

- Confirm the request identifier is returned unchanged for traceability.
- Validate required fields and types; use `payment_fraud_pov.api_contract.validate_response` for basic response-shape checks.
- Check that signal values are in the documented range and missing values remain distinguishable from valid scores.
- Confirm event time, signal timestamp, timezone, and freshness rules with the provider.
- Measure response latency and timeout rates against customer-agreed service targets.
- Test unknown, partial, unavailable, malformed, and delayed responses.
- Validate transaction-key join coverage and duplicate handling in offline analysis.
- Agree on workflow behavior when signals are missing or late; do not infer a production fallback from this demo.
- Confirm logging, retention, access control, encryption, and data-minimization requirements with the responsible teams.

The response validator checks field shape and numeric ranges only. It does not test network behavior, authentication, freshness, service-level objectives, or the correctness of a signal.
