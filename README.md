# OBELISK

## Setup

Virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Instantiate local configuration:

```bash
cp etc/obelisk.toml.tpl etc/obelisk.toml
cp etc/tidb.toml.tpl etc/tidb.toml
chmod 600 etc/obelisk.toml
```

Open `etc/obelisk.toml` and fill local values:

```toml
[llm]
api_key = "<your-llm-api-key>"
# Optional for OpenAI-compatible providers.
base_url = "https://llm.example.com/v1"
```

Also set local TiDB credentials, workload paths, and run parameters in that same
file. Never put secrets in `*.toml.tpl`; templates are tracked, while local
`etc/*.toml` files are ignored.

If `run.repository_name` is left empty, OBELISK uses one plan repository under
`cache/` per configured database schema. Override it only when intentionally
isolating an experiment.

## Run

```bash
./run.sh
```
