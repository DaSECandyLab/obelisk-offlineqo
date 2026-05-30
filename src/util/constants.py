DEFAULT_TIMEOUT_MS: int = 200_000
MEM_QUOTA_BYTES: int = 16 * 1024 * 1024 * 1024  

COST_FACTOR_DOC: dict[str, dict[str, str]] = {
    "tidb_opt_hash_agg_cost_factor": {
        "description": "a float number to control the cost of Hash Aggregation operator."
    },
    "tidb_opt_hash_join_cost_factor": {
        "description": "a float number to control the cost of Hash Join operator."
    },
    "tidb_opt_index_join_cost_factor": {
        "description": "a float number to control the cost of Index Join operator."
    },
    "tidb_opt_index_lookup_cost_factor": {
        "description": "a float number to control the cost of Index Lookup operator."
    },
    "tidb_opt_index_reader_cost_factor": {
        "description": "a float number to control the cost of Index Reader operator."
    },
    "tidb_opt_index_scan_cost_factor": {
        "description": "a float number to control the cost of Index Scan operator."
    },
    "tidb_opt_limit_cost_factor": {
        "description": "a float number to control the cost of Limit operator."
    },
    "tidb_opt_merge_join_cost_factor": {
        "description": "a float number to control the cost of Merge Join operator."
    },
    "tidb_opt_sort_cost_factor": {
        "description": "a float number to control the cost of Sort operator."
    },
    "tidb_opt_stream_agg_cost_factor": {
        "description": "a float number to control the cost of Stream Aggregation operator."
    },
    "tidb_opt_table_full_scan_cost_factor": {
        "description": "a float number to control the cost of Table Full Scan operator."
    },
    "tidb_opt_table_range_scan_cost_factor": {
        "description": "a float number to control the cost of Table Range Scan operator."
    },
    "tidb_opt_table_reader_cost_factor": {
        "description": "a float number to control the cost of Table Reader operator."
    },
    "tidb_opt_table_rowid_scan_cost_factor": {
        "description": "a float number to control the cost of Table RowID Scan operator."
    },
    "tidb_opt_table_tiflash_scan_cost_factor": {
        "description": "a float number to control the cost of Table TiFlash Scan operator."
    },
    "tidb_opt_topn_cost_factor": {
        "description": "a float number to control the cost of TopN operator."
    }
}
