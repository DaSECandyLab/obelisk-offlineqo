[database]
host = "127.0.0.1"
port = 4000
user = "root"
password = ""
name = "imdb"
ca_path = ""
autocommit = true
mem_quota_bytes = 17179869184
validate_copr_cache = true

[llm]
enabled = true
model_name = "gpt-4.1-mini"
# Optional OpenAI-compatible endpoint.
base_url = ""
# Copy this template to etc/obelisk.toml, then fill your local key there.
# Never commit a real LLM API key to a *.tpl file.
api_key = ""
temperature = 0.7
max_retries = 5
retry_delay = 5
max_new_tokens = 2048
top_p = 0.7
prompt_optimizer_enabled = false
prompt_optimizer_iterations = 1
prompt_optimizer_top_n = 3

[run]
sql_dir = "sql/job"
results_dir = "results/job"
# Paper default budget: 6 Sobol warm-start rounds plus 15 optimization rounds.
total_rounds = 21
warm_start_rounds = 6
strategy = "tcbo"
# Empty means one plan repository per database schema, as in the OBELISK paper.
repository_name = ""

[optimization]
baseline_timeout_ms = 3600000
timeout_multiplier = 2.0
topk = 5
batch = 5
retry_attempts = 8
max_no_improvement = 3
tcbo_num_trust_regions = 4
tcbo_risk_threshold = 0.05
tcbo_candidate_count = 2000
