# Security Policy

## Supported branch

Security fixes are applied to `main`.

## Reporting

Do not open a public issue containing credentials, private prompts, database
contents, API keys, or exploitable details. Report the issue privately to the
repository owner with:

- the affected commit;
- reproduction steps;
- impact;
- logs with secrets removed;
- a proposed mitigation when available.

## Production requirements

- Set `ELITE_ADMIN_TOKEN` to a random value of at least 24 characters.
- Set `ELITE_ALLOWED_ORIGINS` to the exact deployed frontend origins.
- Set `ELITE_ALLOWED_HOSTS` when the deployment hostnames are stable.
- Never commit `.env`, SQLite databases, model traces, benchmark responses, or
  user uploads.
- Rotate a credential immediately after suspected exposure.
- Run `python scripts/release_gate_v30.py` before release.
- Run the live V30 coding benchmark and load test against the deployment before
  assigning a 10/10 readiness rating.
