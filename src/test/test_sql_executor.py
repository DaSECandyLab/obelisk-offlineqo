#!/usr/bin/env python3
"""Unit tests for SQLExecutor error fallbacks."""

# ruff: noqa: E402

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pymysql

CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from db.sql_executor import SQLExecutor
from optimization.evaluator import EvaluationStatus


class TestSQLExecutor(unittest.TestCase):
    def test_repository_query_key_preserves_literals_for_exact_duplicate_reuse(self) -> None:
        executor = object.__new__(SQLExecutor)

        with tempfile.TemporaryDirectory() as tmpdir:
            q1 = Path(tmpdir) / "q1.sql"
            q2 = Path(tmpdir) / "nested" / "q2.sql"
            q2.parent.mkdir()
            q1.write_text("SELECT * FROM t WHERE a = 1 AND b = 'x';", encoding="utf-8")
            q2.write_text("select * from t where a = 99 and b = 'y'", encoding="utf-8")

            self.assertNotEqual(
                executor._repository_query_key(str(q1)),
                executor._repository_query_key(str(q2)),
            )

    def test_repository_query_key_normalizes_formatting_only(self) -> None:
        executor = object.__new__(SQLExecutor)

        self.assertEqual(
            executor._normalize_sql_statement_key(
                "SELECT  *\nFROM t WHERE a = 1 /* comment */;"
            ),
            "SELECT * FROM t WHERE a = 1",
        )

    def test_relevant_knobs_falls_back_to_cost_factors(self) -> None:
        executor = object.__new__(SQLExecutor)
        executor._load_sql_from_file = lambda _path: "SELECT 1"

        def execute(query):
            if query.startswith("EXPLAIN"):
                raise pymysql.err.OperationalError(
                    1791,
                    "Unknown EXPLAIN format name: 'relevant_knobs'",
                )
            self.assertIn("SHOW VARIABLES WHERE", query)
            return [
                {"Variable_name": "tidb_opt_hash_join_cost_factor", "Value": "1"},
                {"Variable_name": "tidb_opt_sort_cost_factor", "Value": "2"},
            ]

        executor._execute_query = execute
        executor.execute_query = execute

        knobs = executor.get_relevant_knobs("q.sql")

        self.assertEqual([knob["var"] for knob in knobs], [
            "tidb_opt_hash_join_cost_factor",
            "tidb_opt_sort_cost_factor",
        ])
        self.assertEqual(knobs[0]["min"], 0.1)
        self.assertEqual(knobs[0]["max"], 10.0)
        self.assertEqual(knobs[1]["min"], 0.2)
        self.assertEqual(knobs[1]["max"], 20.0)

    def test_relevant_knobs_falls_back_to_txsql_cost_variables(self) -> None:
        executor = object.__new__(SQLExecutor)
        executor._load_sql_from_file = lambda _path: "SELECT 1"

        def execute(query):
            if query.startswith("EXPLAIN"):
                raise pymysql.err.OperationalError(
                    1791,
                    "Unknown EXPLAIN format name: 'relevant_knob'",
                )
            return [
                {"Variable_name": "txsql_index_pruning_cost_ratio", "Value": "3"},
                {"Variable_name": "txsql_parallel_cost_threshold", "Value": "50000"},
                {"Variable_name": "txsql_spm_auto_capture_cost_threshold", "Value": "0"},
            ]

        executor._execute_query = execute
        executor.execute_query = execute

        knobs = executor.get_relevant_knobs("q.sql")

        self.assertEqual([knob["var"] for knob in knobs], [
            "txsql_index_pruning_cost_ratio",
            "txsql_parallel_cost_threshold",
        ])
        self.assertEqual(knobs[0]["min"], 0.3)
        self.assertEqual(knobs[0]["max"], 30.0)
        self.assertEqual(knobs[1]["min"], 5000.0)
        self.assertEqual(knobs[1]["max"], 500000.0)

    def test_relevant_knobs_accepts_paper_singular_format(self) -> None:
        executor = object.__new__(SQLExecutor)
        executor._load_sql_from_file = lambda _path: "SELECT 1"
        queries = []

        def execute(query):
            queries.append(query)
            if "relevant_knobs" in query:
                raise pymysql.err.OperationalError(
                    1791,
                    "Unknown EXPLAIN format name: 'relevant_knobs'",
                )
            return [
                {
                    "relevant_knob": (
                        '[{"var": "tidb_opt_hash_join_cost_factor", '
                        '"max": 8.0, "default": 1.0}]'
                    )
                }
            ]

        executor._execute_query = execute

        knobs = executor.get_relevant_knobs("q.sql")

        self.assertIn("FORMAT='relevant_knob'", queries[-1])
        self.assertEqual(knobs, [
            {
                "var": "tidb_opt_hash_join_cost_factor",
                "max": 8.0,
                "default": 1.0,
                "min": 0.125,
            }
        ])

    def test_relevant_knobs_extracts_json_array_from_unknown_column_name(self) -> None:
        result = [{
            "Relevant Knobs": (
                '[{"var": "tidb_opt_hash_join_cost_factor", '
                '"min": 0.1, "max": 10.0, "default": 1.0}]'
            )
        }]

        knobs_json = SQLExecutor._extract_relevant_knobs_json(result)

        self.assertIn("tidb_opt_hash_join_cost_factor", knobs_json)

    def test_build_explain_sql_injects_hint_into_select_query(self) -> None:
        executor = object.__new__(SQLExecutor)

        sql = executor._build_explain_sql(
            "SELECT * FROM t",
            {"tidb_opt_hash_join_cost_factor": 0.5000000000000001},
            max_execution_time_ms=1000,
        )

        self.assertIn("EXPLAIN ANALYZE format='tidb_json' SELECT /*+", sql)
        self.assertIn("set_var(tidb_opt_hash_join_cost_factor='0.5')", sql)
        self.assertNotIn("0.5000000000000001", sql)
        self.assertIn("MAX_EXECUTION_TIME(1000)", sql)
        self.assertNotIn("SELECT /*+", sql.split("format='tidb_json'", 1)[0])

    def test_build_explain_sql_preserves_cte_before_top_level_select(self) -> None:
        executor = object.__new__(SQLExecutor)

        sql = executor._build_explain_sql(
            "WITH q AS (SELECT * FROM t) SELECT * FROM q",
            {"tidb_opt_hash_join_cost_factor": 0.5},
            max_execution_time_ms=0,
        )

        self.assertTrue(sql.startswith("EXPLAIN format='tidb_json' WITH q AS"))
        self.assertIn(") SELECT /*+ set_var(tidb_opt_hash_join_cost_factor='0.5') */ * FROM q", sql)

    def test_build_hinted_sql_returns_reproducible_select_without_explain(self) -> None:
        executor = object.__new__(SQLExecutor)

        with tempfile.TemporaryDirectory() as tmpdir:
            sql_path = Path(tmpdir) / "q.sql"
            sql_path.write_text(
                "WITH q AS (SELECT * FROM t) SELECT * FROM q;",
                encoding="utf-8",
            )

            sql = executor.build_hinted_sql(
                str(sql_path),
                {
                    "tidb_opt_hash_join_cost_factor": 100000.00000000001,
                    "tidb_join_order_cost_factor:t1,t2": 3.0000000000000004,
                },
            )

        self.assertTrue(sql.startswith("WITH q AS"))
        self.assertNotIn("EXPLAIN", sql)
        self.assertIn(") SELECT /*+ set_var(tidb_opt_hash_join_cost_factor='100000')", sql)
        self.assertNotIn("100000.00000000001", sql)
        self.assertIn("set_var(tidb_opt_join_order_cost_factor='", sql)
        self.assertIn('"t1,t2": "3"', sql)

    def test_get_explain_result_accepts_case_insensitive_tidb_json_column(self) -> None:
        executor = object.__new__(SQLExecutor)
        executor._load_sql_from_file = lambda _path: "SELECT 1"
        executor._build_explain_sql = lambda *_args, **_kwargs: "EXPLAIN format='tidb_json' SELECT 1"
        executor._execute_query = lambda _sql: [{
            "tidb_json": '[{"operatorInfo": "root, plan_a", "executeInfo": "time:1ms"}]'
        }]

        explain_result, timeout = executor._get_explain_result("query.sql", {}, 0)

        self.assertFalse(timeout)
        self.assertEqual(explain_result[0]["operatorInfo"], "root, plan_a")

    def test_get_explain_result_falls_back_to_text_explain_analyze(self) -> None:
        executor = object.__new__(SQLExecutor)
        executor._load_sql_from_file = lambda _path: "SELECT 1"
        queries = []

        def execute(query):
            queries.append(query)
            if "tidb_json" in query:
                raise pymysql.err.OperationalError(
                    1791,
                    "Unknown EXPLAIN format name: 'tidb_json'",
                )
            return [{
                "EXPLAIN": (
                    "-> Rows fetched before execution  (cost=0.00..0.00 rows=1) "
                    "(actual time=0.123..0.456 rows=1 loops=1)\n\n"
                    "QUERY_PLAN_ID: 0x0021a95dbeaabd6e36\n"
                    "QUERY_DIGEST: d1b44b0c19af710b5a679907e284acd2ddc285201794bc69a2389d77baedddae"
                )
            }]

        executor._execute_query = execute

        explain_result, timeout = executor._get_explain_result(
            "query.sql",
            {"tidb_opt_hash_join_cost_factor": 0.5},
            1000,
        )

        self.assertFalse(timeout)
        self.assertTrue(queries[0].startswith("EXPLAIN ANALYZE format='tidb_json'"))
        self.assertTrue(queries[1].startswith("EXPLAIN ANALYZE SELECT /*+"))
        self.assertIn("MAX_EXECUTION_TIME(1000)", queries[1])
        self.assertEqual(
            explain_result[0]["operatorInfo"],
            "Rows fetched before execution, 0x0021a95dbeaabd6e36",
        )
        self.assertEqual(explain_result[0]["executeInfo"], "time:0.456ms, loops:1")
        self.assertEqual(executor._get_plan_fingerprint(explain_result[0]), "0x0021a95dbeaabd6e36")

        executor._get_explain_result("query.sql", {}, 0)
        self.assertEqual(sum("tidb_json" in query for query in queries), 1)

    def test_text_explain_to_operator_preserves_tree_and_node_metrics(self) -> None:
        executor = object.__new__(SQLExecutor)
        explain_text = (
            "-> Limit: 1 row(s)  (cost=420442.75 rows=1) "
            "(actual time=20509.530..20509.530 rows=1 loops=1)\n"
            "    -> Aggregate: count(0)  (cost=420442.75 rows=1) "
            "(actual time=20509.529..20509.529 rows=1 loops=1)\n"
            "        -> Filter: (store_sales.ss_sold_date_sk is not null)  "
            "(cost=281031.85 rows=1394109) "
            "(actual time=0.039..20394.231 rows=2750311 loops=1)\n"
            "            -> Covering index range scan on store_sales using ss_d "
            "over (NULL < ss_sold_date_sk)  (cost=281031.85 rows=1394109) "
            "(actual time=0.038..20196.652 rows=2750311 loops=1)\n\n"
            "QUERY_PLAN_ID: 0x00a74dc3503ea4152e\n"
            "QUERY_DIGEST: 64444321447d0cd8e709506e604b991f2994045deef54753ca7b697eef480938"
        )

        root = executor._text_explain_to_operator(explain_text)
        aggregate = root["subOperators"][0]
        filter_node = aggregate["subOperators"][0]
        scan_node = filter_node["subOperators"][0]

        self.assertEqual(root["id"], "Limit_text")
        self.assertEqual(root["estRows"], 1.0)
        self.assertEqual(root["actRows"], 1.0)
        self.assertEqual(root["executeInfo"], "time:20509.53ms, loops:1")
        self.assertEqual(executor._get_plan_fingerprint(root), "0x00a74dc3503ea4152e")
        self.assertEqual(scan_node["id"], "CoveringIndexRangeScan_text")
        self.assertEqual(scan_node["taskType"], "cop[tikv]")
        self.assertEqual(scan_node["accessObject"], "store_sales")
        self.assertEqual(scan_node["estRows"], 1394109.0)
        self.assertEqual(scan_node["actRows"], 2750311.0)

    def test_text_explain_fallback_uses_tree_without_execution_for_admission(self) -> None:
        executor = object.__new__(SQLExecutor)
        executor._load_sql_from_file = lambda _path: "SELECT 1"
        queries = []

        def execute(query):
            queries.append(query)
            if "tidb_json" in query:
                raise pymysql.err.OperationalError(
                    1791,
                    "Unknown EXPLAIN format name: 'tidb_json'",
                )
            return [{
                "EXPLAIN": (
                    "-> Rows fetched before execution  (cost=0.00..0.00 rows=1)\n\n"
                    "QUERY_PLAN_ID: 0x0021a95dbeaabd6e36"
                )
            }]

        executor._execute_query = execute

        explain_result, timeout = executor._get_explain_result("query.sql", {}, 0)

        self.assertFalse(timeout)
        self.assertTrue(queries[1].startswith("EXPLAIN FORMAT=TREE SELECT 1"))
        self.assertEqual(explain_result[0]["executeInfo"], "")
        self.assertEqual(executor._get_plan_fingerprint(explain_result[0]), "0x0021a95dbeaabd6e36")

    def test_repository_update_records_text_explain_subplans(self) -> None:
        executor = object.__new__(SQLExecutor)
        recorded = []

        class FakeRepository:
            def record_executed_plan(
                self,
                plan_fingerprint,
                query_template,
                execution_time,
                verbose_plan,
            ):
                recorded.append((
                    plan_fingerprint,
                    query_template,
                    execution_time,
                    verbose_plan["id"],
                ))

        executor.plan_repository = FakeRepository()
        explain_text = (
            "-> Aggregate: count(0)  (cost=10.00 rows=1) "
            "(actual time=3.000..3.000 rows=1 loops=1)\n"
            "    -> Covering index range scan on store_sales using ss_d "
            "over (NULL < ss_sold_date_sk)  (cost=9.00 rows=100) "
            "(actual time=0.100..2.000 rows=100 loops=1)\n\n"
            "QUERY_PLAN_ID: 0x00a74dc3503ea4152e"
        )
        plan = [executor._text_explain_to_operator(explain_text)]

        executor._update_plan_repository(100.0, "query.sql", plan)

        self.assertEqual([item[3] for item in recorded], [
            "Aggregate_text",
            "CoveringIndexRangeScan_text",
        ])
        self.assertEqual(recorded[0][0], "0x00a74dc3503ea4152e")
        self.assertTrue(recorded[1][0].startswith("fp_"))

    def test_plan_fingerprint_prefers_explicit_kernel_digest(self) -> None:
        executor = object.__new__(SQLExecutor)
        digest = "84cf0a490cb12362c11bb03aa58784b5181fe781741164dbf9f78ba47f62c48a"

        plan_fingerprint = executor._get_plan_fingerprint({
            "id": "Selection_1",
            "taskType": "cop[tikv]",
            "operatorInfo": f"eq(t.a, 1), {digest}",
        })

        self.assertEqual(plan_fingerprint, digest)

    def test_canonical_plan_fingerprint_normalizes_literals_and_operator_ids(self) -> None:
        executor = object.__new__(SQLExecutor)
        plan_a = {
            "id": "Selection_1",
            "taskType": "cop[tikv]",
            "operatorInfo": "eq(t.a, 1)",
            "subOperators": [
                {
                    "id": "TableFullScan_2",
                    "taskType": "cop[tikv]",
                    "accessObject": "table:t",
                    "operatorInfo": "keep order:false",
                }
            ],
        }
        plan_b = {
            "id": "Selection_99",
            "taskType": "cop[tikv]",
            "operatorInfo": "eq(t.a, 999)",
            "subOperators": [
                {
                    "id": "TableFullScan_100",
                    "taskType": "cop[tikv]",
                    "accessObject": "table:t",
                    "operatorInfo": "keep order:false",
                }
            ],
        }

        fingerprint_a = executor._get_plan_fingerprint(plan_a)
        fingerprint_b = executor._get_plan_fingerprint(plan_b)

        self.assertTrue(fingerprint_a.startswith("fp_"))
        self.assertEqual(fingerprint_a, fingerprint_b)

    def test_canonical_plan_fingerprint_changes_with_operator_tree(self) -> None:
        executor = object.__new__(SQLExecutor)
        scan_plan = {
            "id": "TableFullScan_1",
            "taskType": "cop[tikv]",
            "accessObject": "table:t",
            "operatorInfo": "keep order:false",
        }
        selection_plan = {
            "id": "Selection_1",
            "taskType": "cop[tikv]",
            "operatorInfo": "eq(t.a, 1)",
            "subOperators": [scan_plan],
        }

        self.assertNotEqual(
            executor._get_plan_fingerprint(scan_plan),
            executor._get_plan_fingerprint(selection_plan),
        )

    def test_execute_with_timeout_raises_non_timeout_mysql_error(self) -> None:
        executor = object.__new__(SQLExecutor)
        executor._timeout_threshold_ms = None

        def fail(*_args, **_kwargs):
            raise pymysql.MySQLError("connection lost")

        executor._get_explain_result = fail

        with self.assertRaisesRegex(pymysql.MySQLError, "connection lost"):
            executor.execute_with_timeout(
                "missing.sql",
                {},
                timeout_ms=100,
            )

    def test_execute_with_timeout_records_explicit_timeout_as_tau(self) -> None:
        executor = object.__new__(SQLExecutor)
        executor._timeout_threshold_ms = None
        repository_updates = []
        executor._get_explain_result = lambda *_args, **_kwargs: (
            [{"operatorInfo": "root, plan_timeout", "executeInfo": "time:100ms"}],
            True,
        )
        executor._update_plan_repository = lambda *args: repository_updates.append(args)

        plan_id, execution_time = executor.execute_with_timeout(
            "query.sql",
            {},
            timeout_ms=100,
        )

        self.assertEqual(plan_id, "plan_timeout")
        self.assertEqual(execution_time, 100.0)
        self.assertEqual(len(repository_updates), 1)

    def test_execute_with_timeout_result_returns_timeout_label(self) -> None:
        executor = object.__new__(SQLExecutor)
        executor._timeout_threshold_ms = None
        executor._get_explain_result = lambda *_args, **_kwargs: (
            [{"operatorInfo": "root, plan_timeout", "executeInfo": "time:100ms"}],
            True,
        )
        executor._update_plan_repository = lambda *_args: None

        plan_id, execution_time, timed_out = executor.execute_with_timeout_result(
            "query.sql",
            {},
            timeout_ms=100,
        )

        self.assertEqual(plan_id, "plan_timeout")
        self.assertEqual(execution_time, 100.0)
        self.assertTrue(timed_out)

    def test_execute_with_timeout_result_marks_equal_latency_completed_when_not_censored(self) -> None:
        executor = object.__new__(SQLExecutor)
        executor._timeout_threshold_ms = None
        executor._get_explain_result = lambda *_args, **_kwargs: (
            [{"operatorInfo": "root, plan_a", "executeInfo": "time:100ms"}],
            False,
        )
        executor._update_plan_repository = lambda *_args: None

        _plan_id, execution_time, timed_out = executor.execute_with_timeout_result(
            "query.sql",
            {},
            timeout_ms=100,
        )

        self.assertEqual(execution_time, 100.0)
        self.assertFalse(timed_out)

    def test_execute_with_timeout_records_mysql_timeout_as_tau(self) -> None:
        executor = object.__new__(SQLExecutor)
        executor._timeout_threshold_ms = None

        def fail(*_args, **_kwargs):
            raise pymysql.err.OperationalError(
                3024,
                "Query execution was interrupted, maximum statement execution time exceeded",
            )

        executor._get_explain_result = fail

        plan_id, execution_time = executor.execute_with_timeout(
            "query.sql",
            {},
            timeout_ms=100,
        )

        self.assertEqual(plan_id, "")
        self.assertEqual(execution_time, 100.0)

    def test_execute_with_timeout_rejects_missing_completed_latency(self) -> None:
        executor = object.__new__(SQLExecutor)
        executor._timeout_threshold_ms = 321.0
        executor._get_explain_result = lambda *_args, **_kwargs: (
            [{"operatorInfo": "root, plan_a"}],
            False,
        )

        with self.assertRaisesRegex(ValueError, "root latency could not be parsed"):
            executor.execute_with_timeout(
                "query.sql",
                {},
                timeout_ms=100,
            )

    def test_set_timeout_threshold_replaces_tau_for_each_query(self) -> None:
        executor = object.__new__(SQLExecutor)
        executor._timeout_threshold_ms = None

        self.assertEqual(executor.set_timeout_threshold(100.0), 100.0)
        self.assertEqual(executor.timeout_threshold_ms, 100.0)
        self.assertEqual(executor.set_timeout_threshold(250.0), 250.0)
        self.assertEqual(executor.timeout_threshold_ms, 250.0)

    def test_repository_update_records_only_nodes_with_real_runtime_metrics(self) -> None:
        executor = object.__new__(SQLExecutor)
        recorded = []

        class FakeRepository:
            def record_executed_plan(
                self,
                plan_fingerprint,
                query_template,
                execution_time,
                verbose_plan,
            ):
                recorded.append((
                    plan_fingerprint,
                    query_template,
                    execution_time,
                    verbose_plan["id"],
                ))

        executor.plan_repository = FakeRepository()
        plan = [{
            "id": "HashJoin_1",
            "taskType": "root",
            "executeInfo": "time:5ms, loops:1",
            "operatorInfo": "join, plan_root",
            "subOperators": [
                {
                    "id": "Selection_2",
                    "taskType": "cop[tikv]",
                    "executeInfo": "tikv_task:{proc max:2ms, min:0s}",
                    "operatorInfo": "selection, plan_cop",
                },
                {
                    "id": "Projection_3",
                    "taskType": "root",
                    "operatorInfo": "projection, plan_missing_time",
                },
                {
                    "id": "Projection_4",
                    "taskType": "root",
                    "executeInfo": "time:0s, loops:1",
                    "operatorInfo": "projection, plan_zero_time",
                },
            ],
        }]

        executor._update_plan_repository(100.0, "query.sql", plan)

        self.assertEqual(recorded, [
            ("plan_root", "query.sql", 5.0, "HashJoin_1"),
            ("plan_cop", "query.sql", 2.0, "Selection_2"),
        ])

    def test_subplan_match_stops_at_largest_top_down_match(self) -> None:
        executor = object.__new__(SQLExecutor)

        class FakeRepository:
            def get_execution_time_and_verbose_plan(self, plan_id, _query_template):
                if plan_id in {"plan_parent", "plan_child"}:
                    return 10.0, {"operatorInfo": f"cached, {plan_id}"}
                return None, None

        executor.plan_repository = FakeRepository()
        plan = {
            "id": "HashJoin_1",
            "operatorInfo": "root, plan_root",
            "subOperators": [
                {
                    "id": "TableReader_2",
                    "operatorInfo": "parent, plan_parent",
                    "subOperators": [
                        {
                            "id": "TableScan_3",
                            "operatorInfo": "child, plan_child",
                        }
                    ],
                }
            ],
        }

        matches = executor._subplan_match(plan, "query.sql")

        self.assertEqual([match["plan_fingerprint"] for match in matches], ["plan_parent"])

    def test_subplan_match_returns_all_history_for_largest_match(self) -> None:
        executor = object.__new__(SQLExecutor)
        calls = []

        class FakeRepository:
            def get_matching_plan_records(self, plan_id, query_template=None):
                calls.append((plan_id, query_template))
                if plan_id == "plan_parent":
                    return [
                        (10.0, {"operatorInfo": "cached old, plan_parent"}),
                        (20.0, {"operatorInfo": "cached new, plan_parent"}),
                    ]
                if plan_id == "plan_child":
                    return [(99.0, {"operatorInfo": "cached child, plan_child"})]
                return []

        executor.plan_repository = FakeRepository()
        plan = {
            "id": "HashJoin_1",
            "operatorInfo": "root, plan_root",
            "subOperators": [
                {
                    "id": "TableReader_2",
                    "operatorInfo": "parent, plan_parent",
                    "subOperators": [
                        {
                            "id": "TableScan_3",
                            "operatorInfo": "child, plan_child",
                        }
                    ],
                }
            ],
        }

        matches = executor._subplan_match(plan, "query.sql")

        self.assertEqual(
            [match["historical_latency"] for match in matches],
            [10.0, 20.0],
        )
        self.assertEqual([match["plan_fingerprint"] for match in matches], ["plan_parent", "plan_parent"])
        self.assertIn(("plan_parent", None), calls)

    def test_admission_reject_returns_conservative_subplan_estimate(self) -> None:
        executor = object.__new__(SQLExecutor)
        executor._timeout_threshold_ms = 200.0
        executor._get_explain_result = lambda *_args, **_kwargs: (
            [{"operatorInfo": "root, plan_a"}],
            False,
        )
        executor._subplan_match = lambda *_args, **_kwargs: [
            {"current_operator": {"taskType": "root"}, "historical_plan": {"taskType": "root"}}
        ]

        class FakeRepository:
            def get_execution_time(self, *_args):
                return None

        executor.plan_repository = FakeRepository()

        with patch("db.sql_executor.estimate_subplan_execution_time", return_value=350.0):
            plan_id, execution_time, status = executor.admission_check(
                "query.sql",
                {},
                timeout_ms=200.0,
            )

        self.assertEqual(plan_id, "plan_a")
        self.assertEqual(execution_time, 350.0)
        self.assertEqual(status, EvaluationStatus.SUBPLAN_REJECTED)

    def test_admission_reject_uses_max_estimate_and_rejects_at_threshold(self) -> None:
        executor = object.__new__(SQLExecutor)
        executor._timeout_threshold_ms = 200.0
        executor._get_explain_result = lambda *_args, **_kwargs: (
            [{"operatorInfo": "root, plan_a"}],
            False,
        )
        executor._subplan_match = lambda *_args, **_kwargs: [
            {
                "current_operator": {"taskType": "root"},
                "historical_plan": {"taskType": "root", "id": "old"},
            },
            {
                "current_operator": {"taskType": "root"},
                "historical_plan": {"taskType": "root", "id": "new"},
            },
        ]

        class FakeRepository:
            def get_execution_time(self, *_args):
                return None

        executor.plan_repository = FakeRepository()

        with patch(
            "db.sql_executor.estimate_subplan_execution_time",
            side_effect=[150.0, 200.0],
        ):
            plan_id, execution_time, status = executor.admission_check(
                "query.sql",
                {},
                timeout_ms=200.0,
            )

        self.assertEqual(plan_id, "plan_a")
        self.assertEqual(execution_time, 200.0)
        self.assertEqual(status, EvaluationStatus.SUBPLAN_REJECTED)

    def test_full_plan_duplicate_reuse_requires_same_sql_statement_key(self) -> None:
        executor = object.__new__(SQLExecutor)
        executor._timeout_threshold_ms = 200.0
        executor._get_explain_result = lambda *_args, **_kwargs: (
            [{"operatorInfo": "root, plan_same"}],
            False,
        )
        executor._subplan_match = lambda *_args, **_kwargs: []

        with tempfile.TemporaryDirectory() as tmpdir:
            q1 = Path(tmpdir) / "q1.sql"
            q2 = Path(tmpdir) / "q2.sql"
            q1.write_text("SELECT * FROM t WHERE a = 1;", encoding="utf-8")
            q2.write_text("SELECT * FROM t WHERE a = 99;", encoding="utf-8")
            q1_key = executor._repository_query_key(str(q1))

            class FakeRepository:
                def get_execution_time(self, plan_fingerprint, query_template):
                    if plan_fingerprint == "plan_same" and query_template == q1_key:
                        return 123.0
                    return None

            executor.plan_repository = FakeRepository()

            plan_fingerprint, execution_time, status = executor.admission_check(
                str(q2),
                {},
                timeout_ms=200.0,
            )

        self.assertEqual(plan_fingerprint, "plan_same")
        self.assertIsNone(execution_time)
        self.assertEqual(status, EvaluationStatus.ADMITTED)


if __name__ == "__main__":
    unittest.main(verbosity=2)
