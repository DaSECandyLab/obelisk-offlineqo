import hashlib
import json
import re
from typing import Any, Dict, List, Tuple, Union
import pymysql
from pymysql.cursors import DictCursor
from optimization.evaluator import EvaluationStatus
from util.logger import logger
from db.plan_repository import PlanRepository
from db.subplan import estimate_subplan_execution_time


class SQLExecutor:
    """Executor for running SQL queries and extracting execution metrics."""

    def __init__(
        self,
        cursor: DictCursor,
        repository_name: str = "default",
        *,
        cache_name: str | None = None,
    ) -> None:
        """Initialize SQLExecutor with a database cursor.
        
        Args:
            cursor: Database cursor for executing queries
            repository_name: Plan repository name under cache/<name>.db
        """
        self.cursor = cursor
        self._connection = cursor.connection
        self.knob_space = None
        self._timeout_threshold_ms = None
        self._tidb_json_explain_supported: bool | None = None
        selected_repository = cache_name if cache_name is not None else repository_name
        self.plan_repository = PlanRepository(repository_name=selected_repository)
        # Backward-compatible attribute for older scripts.
        self.cache_manager = self.plan_repository

    def _ensure_connection(self) -> None:
        """Ensure database connection is alive and reconnect if necessary."""
        try:
            self._connection.ping(reconnect=True)
        except Exception as e:
            logger.warning(f"Database connection lost, attempting to reconnect: {str(e)}")
            try:
                self._connection.ping(reconnect=True)
                self.cursor = self._connection.cursor()
            except Exception as e:
                logger.error(f"Failed to reconnect to database: {str(e)}")
                raise

    def execute_query(self, query: str) -> Any:
        """Execute a query with connection check."""
        try:
            return self._execute_query(query)
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            raise

    def _execute_query(self, query: str) -> Any:
        """Execute a query without adding error logs at this layer."""
        self._ensure_connection()
        self.cursor.execute(query)
        return self.cursor.fetchall()

    def _load_sql_from_file(self, filepath: str) -> str:
        """Load SQL query from a file, strip trailing semicolon and whitespace."""
        with open(filepath, 'r', encoding='utf-8') as file:
            return file.read().strip().rstrip(';')

    def _repository_query_key(self, sql_filepath: str) -> str:
        """Return the SQL-statement key Q used for exact duplicate reuse.

        Full-plan duplicate reuse in the paper is scoped by the same SQL
        statement Q.  Literal-generalized reuse is handled separately by
        subplan fingerprints, so this key deliberately preserves literals.
        """
        try:
            sql_query = self._load_sql_from_file(sql_filepath)
        except Exception:
            return sql_filepath
        return self._normalize_sql_statement_key(sql_query)

    @staticmethod
    def _normalize_sql_statement_key(sql_query: str) -> str:
        """Normalize formatting for Q while preserving literal values."""
        text = re.sub(r"/\*.*?\*/", " ", sql_query, flags=re.DOTALL)
        text = re.sub(r"--[^\n]*", " ", text)
        text = re.sub(r"\s+", " ", text).strip().rstrip(";").strip()
        return text

    def get_relevant_knobs(self, sql_filepath: str) -> List[Dict[str, Union[int, float, str]]]:
        """Get tunable knobs for a given SQL query."""
        sql_query = self._load_sql_from_file(sql_filepath)
        if not sql_query:
            return []

        result = None
        last_unknown_format_error = None
        for explain_format in ("relevant_knob", "relevant_knobs"):
            explain_sql = f"EXPLAIN FORMAT='{explain_format}' {sql_query}"
            try:
                result = self._execute_query(explain_sql)
                break
            except pymysql.MySQLError as error:
                if getattr(error, "args", [None])[0] == 1791:
                    last_unknown_format_error = error
                    continue
                raise

        if result is None:
            if last_unknown_format_error is not None:
                return self._get_fallback_cost_factor_knobs()
            return []

        knobs_json = self._extract_relevant_knobs_json(result)
        knobs = json.loads(knobs_json)

        for knob in knobs:
            if "min" not in knob:
                knob['min'] = 1.0 / knob['max']

        return knobs

    @staticmethod
    def _extract_relevant_knobs_json(result: Any) -> str:
        """Extract relevant-knob JSON from TiDB EXPLAIN output."""
        if not result:
            return "[]"

        row = result[0]
        for key, value in row.items():
            normalized_key = key.lower().replace(" ", "_")
            if normalized_key in {"knobs", "relevant_knobs", "relevant_knob"}:
                return value or "[]"
        for value in row.values():
            if not isinstance(value, str):
                continue
            text = value.strip()
            if text.startswith("[") and text.endswith("]"):
                return text
        return "[]"

    def _get_fallback_cost_factor_knobs(self) -> List[Dict[str, Union[int, float, str]]]:
        """Fallback for builds without EXPLAIN FORMAT='relevant_knob'.

        OBELISK kernels expose query-specific C-knobs through relevant_knob.
        Some TiDB/TXSQL-compatible test instances lack that hook but still
        expose optimizer cost variables that can be injected through SET_VAR.
        They are a less precise fallback, kept positive for log-space mapping.
        """
        rows = self.execute_query(
            "SHOW VARIABLES WHERE "
            "Variable_name LIKE 'tidb_opt%cost_factor' "
            "OR Variable_name LIKE 'txsql%cost%' "
            "OR Variable_name = 'secondary_engine_cost_threshold';"
        )
        knobs = []
        for row in rows:
            name = row.get("Variable_name") or row.get("variable_name")
            if not name:
                continue

            raw_default = row.get("Value")
            if raw_default is None:
                raw_default = row.get("value")
            try:
                default_value = float(raw_default)
            except (TypeError, ValueError):
                continue
            if default_value <= 0:
                continue

            knobs.append({
                "var": str(name),
                "min": default_value / 10.0,
                "max": default_value * 10.0,
                "default": default_value,
            })

        if not knobs:
            raise RuntimeError("No positive optimizer cost-variable knobs were found")

        logger.warning(
            "EXPLAIN relevant-knob formats are unavailable; using %d optimizer cost-variable knobs",
            len(knobs),
        )
        return knobs

    def build_hinted_sql(
        self,
        sql_filepath: str,
        knobs: Dict[str, Union[int, float, str]],
        max_execution_time_ms: int = 0,
    ) -> str:
        """Build executable SQL with optimizer hints for a tuned configuration."""
        sql_query = self._load_sql_from_file(sql_filepath)
        if not sql_query:
            raise ValueError(f"Empty SQL in file {sql_filepath}")
        return self._build_hinted_sql(sql_query, knobs, int(max_execution_time_ms))

    @classmethod
    def _build_hinted_sql(
        cls,
        sql_query: str,
        knobs: Dict[str, Union[int, float, str]],
        max_execution_time_ms: int = 0,
    ) -> str:
        """Build a SELECT statement with OBELISK C-knob hints."""
        hint_items = [
            f"set_var({key}='{cls._format_hint_value(value)}')"
            for key, value in knobs.items()
            if not key.startswith("tidb_join_order_cost_factor:")
        ]

        join_order_factors = {
            k: v
            for k, v in knobs.items()
            if k.startswith("tidb_join_order_cost_factor:")
        }
        if len(join_order_factors) > 0:
            join_order_factors = {
                k[len("tidb_join_order_cost_factor:"):]: cls._format_hint_value(v)
                for k, v in join_order_factors.items()
            }
            hint_items.append(
                "set_var(tidb_opt_join_order_cost_factor='%s')"
                % json.dumps(join_order_factors, sort_keys=True)
            )

        if max_execution_time_ms > 0:
            hint_items.append(f"MAX_EXECUTION_TIME({max_execution_time_ms})")

        return cls._inject_optimizer_hint(sql_query, " ".join(hint_items))

    @staticmethod
    def _format_hint_value(value: Union[int, float, str]) -> str:
        if isinstance(value, bool):
            return "1" if value else "0"
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float):
            if not (value == value and value not in (float("inf"), -float("inf"))):
                raise ValueError("Hint value must be finite")
            return f"{value:.12g}"
        return str(value)

    def _build_explain_sql(self, sql_query: str, knobs: Dict[str, Union[int, float, str]],
                           max_execution_time_ms: int = 0) -> str:
        """Build EXPLAIN SQL statement with optimizer hints and optional execution timeout."""
        hinted_sql = self._build_hinted_sql(sql_query, knobs, max_execution_time_ms)
        explain_sql = (
            f"EXPLAIN ANALYZE format='tidb_json' {hinted_sql}"
            if max_execution_time_ms > 0 else
            f"EXPLAIN format='tidb_json' {hinted_sql}"
        )
        return explain_sql

    @classmethod
    def _inject_optimizer_hint(cls, sql_query: str, hint_text: str) -> str:
        """Insert TiDB optimizer hints after the top-level SELECT keyword."""
        if not hint_text:
            return sql_query

        select_pos = cls._find_top_level_select(sql_query)
        if select_pos < 0:
            raise ValueError("Only SELECT queries can be optimized with C-knob hints")

        insert_pos = select_pos + len("select")
        return f"{sql_query[:insert_pos]} /*+ {hint_text} */{sql_query[insert_pos:]}"

    @staticmethod
    def _find_top_level_select(sql_query: str) -> int:
        depth = 0
        quote = ""
        index = 0
        while index < len(sql_query):
            char = sql_query[index]
            nxt = sql_query[index:index + 2]

            if quote:
                if char == quote:
                    quote = ""
                elif char == "\\":
                    index += 1
                index += 1
                continue

            if nxt == "--":
                newline = sql_query.find("\n", index + 2)
                index = len(sql_query) if newline < 0 else newline + 1
                continue
            if nxt == "/*":
                end = sql_query.find("*/", index + 2)
                index = len(sql_query) if end < 0 else end + 2
                continue
            if char in {"'", '"', "`"}:
                quote = char
                index += 1
                continue
            if char == "(":
                depth += 1
                index += 1
                continue
            if char == ")":
                depth = max(0, depth - 1)
                index += 1
                continue
            if depth == 0 and sql_query[index:index + 6].lower() == "select":
                before = sql_query[index - 1] if index > 0 else " "
                after = sql_query[index + 6] if index + 6 < len(sql_query) else " "
                if not (before.isalnum() or before == "_") and not (after.isalnum() or after == "_"):
                    return index
            index += 1

        return -1
    
    
    def _get_explain_result(self, sql_filepath: str, knobs: Dict[str, Union[int, float, str]], timeout_ms: int=0) -> Tuple[List[Dict[str, Any]], bool]:
        """Get explain plan for SQL query with given knobs."""
        sql_query = self._load_sql_from_file(sql_filepath)
        if not sql_query:
            raise ValueError(f"Empty SQL in file {sql_filepath}")
        if getattr(self, "_tidb_json_explain_supported", None) is False:
            return self._get_text_explain_result(sql_query, knobs, int(timeout_ms))

        explain_sql = self._build_explain_sql(sql_query, knobs, int(timeout_ms))
        try:
            result = self._execute_query(explain_sql)
            explain_result = self._extract_tidb_json(result)
            self._tidb_json_explain_supported = True
            return self._parse_tidb_json_explain(explain_result)
        except pymysql.MySQLError as error:
            if not self._is_unsupported_explain_format(error):
                raise
            self._tidb_json_explain_supported = False
            logger.warning("EXPLAIN FORMAT='tidb_json' is unavailable; using text EXPLAIN fallback")
        except (KeyError, ValueError, json.JSONDecodeError) as error:
            self._tidb_json_explain_supported = False
            logger.warning("Failed to parse TiDB JSON EXPLAIN; using text EXPLAIN fallback: %s", error)

        return self._get_text_explain_result(sql_query, knobs, int(timeout_ms))

    @staticmethod
    def _parse_tidb_json_explain(explain_result: str) -> Tuple[List[Dict[str, Any]], bool]:
        timeout_pos = explain_result.rfind(':TIMEOUT')
        timeout = False
        if timeout_pos != -1:
            explain_result = explain_result[:timeout_pos]
            timeout = True
        return json.loads(explain_result), timeout

    @staticmethod
    def _extract_tidb_json(result: Any) -> str:
        """Extract TiDB JSON plan text from a cursor result row."""
        if not result:
            raise ValueError("EXPLAIN returned no rows")

        row = result[0]
        for key, value in row.items():
            if key.lower() == "tidb_json":
                return value
        raise KeyError("EXPLAIN result does not contain TiDB_JSON")

    @staticmethod
    def _is_unsupported_explain_format(error: pymysql.MySQLError) -> bool:
        args = getattr(error, "args", ())
        return (
            bool(args and args[0] == 1791)
            or "unknown explain format" in str(error).lower()
        )

    def _get_text_explain_result(
        self,
        sql_query: str,
        knobs: Dict[str, Union[int, float, str]],
        timeout_ms: int = 0,
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """Fallback for TiDB/TXSQL builds without JSON plan output."""
        explain_sql = self._build_text_explain_sql(sql_query, knobs, timeout_ms)
        result = self._execute_query(explain_sql)
        explain_text = self._extract_text_explain(result)
        return [self._text_explain_to_operator(explain_text)], ":TIMEOUT" in explain_text

    def _build_text_explain_sql(
        self,
        sql_query: str,
        knobs: Dict[str, Union[int, float, str]],
        timeout_ms: int = 0,
    ) -> str:
        hinted_sql = self._build_hinted_sql(sql_query, knobs, timeout_ms)
        if timeout_ms > 0:
            return f"EXPLAIN ANALYZE {hinted_sql}"
        return f"EXPLAIN FORMAT=TREE {hinted_sql}"

    @staticmethod
    def _extract_text_explain(result: Any) -> str:
        if not result:
            raise ValueError("EXPLAIN returned no rows")

        lines = []
        for row in result:
            if not isinstance(row, dict):
                lines.append(str(row))
                continue
            for key, value in row.items():
                if key.lower() == "explain":
                    lines.append(str(value))
                    break
            else:
                lines.append("\t".join(str(value) for value in row.values()))
        return "\n".join(lines)

    @classmethod
    def _text_explain_to_operator(cls, explain_text: str) -> Dict[str, Any]:
        plan_fingerprint = cls._extract_text_plan_fingerprint(explain_text)
        root = cls._parse_text_explain_tree(explain_text)
        if root is None:
            execution_time_ms = cls._extract_text_actual_time_ms(explain_text)
            execute_info = f"time:{execution_time_ms}ms" if execution_time_ms is not None else ""
            root = {
                "id": "TextExplainPlan",
                "taskType": "root",
                "accessObject": "",
                "operatorInfo": "text_explain_plan",
                "executeInfo": execute_info,
                "subOperators": [],
            }

        root["operatorInfo"] = cls._append_text_plan_fingerprint(
            str(root.get("operatorInfo", "") or "text_explain_plan"),
            plan_fingerprint,
        )
        root["textPlan"] = explain_text
        return root

    @classmethod
    def _parse_text_explain_tree(cls, explain_text: str) -> Dict[str, Any] | None:
        """Parse TiDB/TXSQL text EXPLAIN output into a lightweight operator tree."""
        root = None
        stack: list[tuple[int, Dict[str, Any]]] = []
        for raw_line in explain_text.splitlines():
            arrow_pos = raw_line.find("->")
            if arrow_pos < 0:
                continue
            prefix = raw_line[:arrow_pos]
            if prefix.strip():
                continue

            body = raw_line[arrow_pos + 2:].strip()
            if not body:
                continue
            node = cls._text_explain_line_to_operator(body)

            while stack and stack[-1][0] >= arrow_pos:
                stack.pop()
            if stack:
                stack[-1][1].setdefault("subOperators", []).append(node)
            else:
                root = node
            stack.append((arrow_pos, node))
        return root

    @classmethod
    def _text_explain_line_to_operator(cls, line: str) -> Dict[str, Any]:
        cost_match = re.search(
            r"\s+\(cost=(?P<cost>\d+(?:\.\d+)?)(?:\.\.\d+(?:\.\d+)?)?"
            r"\s+rows=(?P<est_rows>\d+(?:\.\d+)?)\)",
            line,
        )
        actual_match = re.search(
            r"\s+\(actual time=(?P<actual_start>\d+(?:\.\d+)?)\.\."
            r"(?P<actual_end>\d+(?:\.\d+)?)\s+rows=(?P<act_rows>\d+(?:\.\d+)?)"
            r"\s+loops=(?P<loops>\d+)\)",
            line,
        )

        description = line
        if cost_match:
            description = description.replace(cost_match.group(0), "")
        if actual_match:
            description = description.replace(actual_match.group(0), "")
        description = re.sub(r"\s+\(never executed\)", "", description).strip()

        node: Dict[str, Any] = {
            "id": cls._text_operator_id(description),
            "taskType": cls._text_task_type(description),
            "accessObject": cls._text_access_object(description),
            "operatorInfo": description,
            "subOperators": [],
        }
        if cost_match:
            node["estRows"] = float(cost_match.group("est_rows"))
            node["estCost"] = float(cost_match.group("cost"))
        if actual_match:
            node["actRows"] = float(actual_match.group("act_rows"))
            node["executeInfo"] = (
                f"time:{float(actual_match.group('actual_end'))}ms, "
                f"loops:{actual_match.group('loops')}"
            )
        else:
            node["executeInfo"] = ""
        return node

    @staticmethod
    def _append_text_plan_fingerprint(operator_info: str, plan_fingerprint: str) -> str:
        if not plan_fingerprint:
            return operator_info
        return f"{operator_info}, {plan_fingerprint}"

    @staticmethod
    def _text_operator_id(description: str) -> str:
        operator_text = description.split(":", 1)[0]
        operator_text = re.split(r"\bon\b|\busing\b|\bover\b", operator_text, flags=re.IGNORECASE)[0]
        words = re.findall(r"[A-Za-z]+", operator_text)
        if not words:
            return "TextOperator"
        return "".join(word.capitalize() for word in words) + "_text"

    @staticmethod
    def _text_task_type(description: str) -> str:
        lower_description = description.lower()
        if (
            "scan" in lower_description
            or "point lookup" in lower_description
            or "index lookup" in lower_description
        ):
            return "cop[tikv]"
        return "root"

    @staticmethod
    def _text_access_object(description: str) -> str:
        match = re.search(r"\bon\s+([`A-Za-z0-9_.]+)", description, flags=re.IGNORECASE)
        if not match:
            return ""
        return match.group(1).strip("`")

    @classmethod
    def _extract_text_plan_fingerprint(cls, explain_text: str) -> str:
        match = re.search(r"QUERY_PLAN_ID:\s*(0x[0-9a-fA-F]+|[0-9a-fA-F]+)", explain_text)
        if match:
            return match.group(1).lower()

        normalized = cls._normalize_text_explain_for_fingerprint(explain_text)
        return "fp_text_" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    @staticmethod
    def _normalize_text_explain_for_fingerprint(explain_text: str) -> str:
        text = re.sub(r"\(actual time=[^)]+\)", "(actual time=?)", explain_text)
        text = re.sub(r"QUERY_DIGEST:\s*[0-9a-fA-F]+", "QUERY_DIGEST: ?", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip().lower()

    @staticmethod
    def _extract_text_actual_time_ms(explain_text: str) -> float | None:
        for line in explain_text.splitlines():
            match = re.search(r"actual time=(\d+(?:\.\d+)?)\.\.(\d+(?:\.\d+)?)", line)
            if match:
                return float(match.group(2))
        return None

    def _get_plan_fingerprint(self, operator: Dict[str, Any]) -> str:
        """Return the plan fingerprint F(P) used by the plan repository.

        Modified OBELISK/TiDB kernels append an explicit digest to
        operatorInfo. When that digest is absent, derive a deterministic
        fingerprint from a literal-normalized operator tree.
        """
        explicit_fingerprint = self._extract_explicit_plan_fingerprint(operator)
        if explicit_fingerprint:
            return explicit_fingerprint
        return self._canonical_plan_fingerprint(operator)

    def _get_plan_id(self, operator: Dict[str, Any]) -> str:
        """Backward-compatible alias for the plan fingerprint F(P)."""
        return self._get_plan_fingerprint(operator)

    @staticmethod
    def _extract_explicit_plan_fingerprint(operator: Dict[str, Any]) -> str:
        operator_info = str(operator.get("operatorInfo", "") or "")
        if not operator_info:
            return ""

        candidate = operator_info.rsplit(",", 1)[-1].strip()
        if not candidate:
            return ""
        if re.fullmatch(r"0x[0-9a-fA-F]{16,}", candidate):
            return candidate.lower()
        if re.fullmatch(r"[0-9a-fA-F]{16,}", candidate):
            return candidate.lower()
        if re.fullmatch(r"plan_[A-Za-z0-9_./:-]+", candidate):
            return candidate
        return ""

    @classmethod
    def _canonical_plan_fingerprint(cls, operator: Dict[str, Any]) -> str:
        canonical = cls._canonical_plan_node(operator)
        payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
        return "fp_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def _canonical_plan_node(cls, operator: Dict[str, Any]) -> Dict[str, Any]:
        children = [
            cls._canonical_plan_node(child)
            for child in operator.get("subOperators", [])
            if isinstance(child, dict)
        ]
        return {
            "operator_type": cls._operator_type(operator.get("id", "")),
            "task_type": operator.get("taskType", ""),
            "access_object": cls._normalize_plan_text(operator.get("accessObject", "")),
            "operator_info": cls._normalize_plan_text(operator.get("operatorInfo", "")),
            "children": children,
        }

    @staticmethod
    def _operator_type(operator_id: Any) -> str:
        text = str(operator_id or "")
        text = re.sub(r"\([^)]*\)$", "", text)
        text = re.sub(r"_\d+$", "", text)
        return text

    @staticmethod
    def _normalize_plan_text(value: Any) -> str:
        text = str(value or "")
        text = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)
        text = re.sub(r"--[^\n]*", " ", text)
        text = re.sub(r"'(?:''|[^'])*'", "?", text)
        text = re.sub(r"\b\d+(?:\.\d+)?\b", "?", text)
        text = re.sub(r"\b0x[0-9a-fA-F]{16,}\b", "?", text)
        text = re.sub(r"\b[0-9a-fA-F]{16,}\b", "?", text)
        return re.sub(r"\s+", " ", text).strip().lower()

    def _extract_time_seconds(self, plan_info: Dict[str, Any]) -> float | None:
        """Extract execution time from info string and convert to milliseconds."""
        execute_info = plan_info.get('executeInfo', '')
        if not execute_info:
            logger.warning("No Execute Info")
            return None
        
        text = execute_info.split(',')[0]
        if not text:
            return None

        match = re.search(r'(?<![a-zA-Z_])time:(\d+(?:\.\d+)?)([a-zµ]+)', text)
        
        if not match:
            if ':' in text:
                last_colon_content = text.split(':')[-1].strip()
                time_match = re.search(r'(\d+(?:\.\d+)?)([a-zµ]+)', last_colon_content)
                if time_match:
                    match = time_match
                else:
                    logger.info(f"No valid time format found in last colon content: {last_colon_content}")
                    return None
            else:
                logger.info(f"No valid time format found in text: {text}")
                return None
        
        value_str, unit = match.groups()
        value = float(value_str)
        unit = unit.lower()
        if unit == 's':
            return value * 1_000
        if unit == 'ms':
            return value
        if unit in ('µs', 'us'):
            return value / 1_000
        if unit == 'm':
            return value * 60 * 1_000
        if unit == 'h':
            return value * 60 * 60 * 1_000
        raise ValueError(f"Unknown time unit: {unit}")
    

    def _update_plan_repository(
        self,
        timeout_ms: float,
        sql_filepath: str,
        explain_result: List[Dict[str, Any]],
    ) -> None:
        """Record actually executed plans and subplans in repository R."""
        query_template = self._repository_query_key(sql_filepath)
        self._record_executed_subplans(explain_result[0], query_template, timeout_ms)

    def _record_executed_subplans(
        self,
        node: Dict[str, Any],
        query_template: str,
        timeout_ms: float,
    ) -> None:
        """Record runtime-annotated plan nodes after execution."""
        plan_fingerprint = self._get_plan_fingerprint(node)
        execution_time = self._extract_time_seconds(node)
        if plan_fingerprint and execution_time is not None and 0 < execution_time < timeout_ms:
            self.plan_repository.record_executed_plan(
                plan_fingerprint,
                query_template,
                execution_time,
                node,
            )

        sub_ops = node.get('subOperators', [])
        for sub_node in sub_ops:
            if isinstance(sub_node, dict):
                self._record_executed_subplans(sub_node, query_template, timeout_ms)

    def _update_cache(
        self,
        timeout_ms: float,
        sql_filepath: str,
        explain_result: List[Dict[str, Any]],
    ) -> None:
        """Backward-compatible wrapper for repository updates."""
        self._update_plan_repository(timeout_ms, sql_filepath, explain_result)


    def execute_with_timeout_result(
        self,
        sql_filepath: str,
        knobs: Dict[str, Union[int, float, str]],
        timeout_ms: int,
    ) -> Tuple[str, float, bool]:
        """Execute SQL and return plan fingerprint, latency, and timeout label o."""
        plan_fingerprint = ""
        try:
            explain_result, timeout = self._get_explain_result(sql_filepath, knobs, timeout_ms)
            plan_fingerprint = self._get_plan_fingerprint(explain_result[0])

            if timeout:
                logger.warning("Execution timeout: maximum statement execution time exceeded")
                self._update_plan_repository(timeout_ms, sql_filepath, explain_result)
                return plan_fingerprint, float(timeout_ms), True
            else:
                execution_time = self._extract_time_seconds(explain_result[0])
                if execution_time is None:
                    raise ValueError(
                        "Execution completed but root latency could not be parsed"
                    )

                self._update_plan_repository(timeout_ms, sql_filepath, explain_result)
                return plan_fingerprint, execution_time, False

        except pymysql.MySQLError as error:
            if self._is_statement_timeout_error(error):
                logger.warning("Execution timeout: maximum statement execution time exceeded")
                return plan_fingerprint, float(timeout_ms), True
            logger.error("Non-timeout MySQL error during execution: %s", error)
            raise

    def execute_with_timeout(self, sql_filepath: str, knobs: Dict[str, Union[int, float, str]],
                             timeout_ms: int) -> Tuple[str, float]:
        """Backward-compatible execution API returning plan fingerprint and latency."""
        plan_fingerprint, execution_time, _ = self.execute_with_timeout_result(
            sql_filepath,
            knobs,
            timeout_ms,
        )
        return plan_fingerprint, execution_time
    

    def _subplan_match(self, operator: Dict[str, Any], query_template: str) -> list:
        """Return largest top-down historical subplan matches in repository R."""
        matches = []
        if not operator:
            return matches
        plan_fingerprint = self._get_plan_fingerprint(operator)
        historical_records = self._matching_plan_records(plan_fingerprint, query_template)
        if historical_records:
            for historical_latency, historical_plan in historical_records:
                matches.append({
                    "plan_fingerprint": plan_fingerprint,
                    "current_operator": operator,
                    "historical_latency": historical_latency,
                    "historical_plan": historical_plan,
                })
            return matches
        sub_ops = operator.get('subOperators', [])
        for sub_op in sub_ops:
            matches.extend(self._subplan_match(sub_op, query_template))
        return matches

    def _matching_plan_records(
        self,
        plan_fingerprint: str,
        query_template: str,
    ) -> list[tuple[float, Dict[str, Any]]]:
        if hasattr(self.plan_repository, "get_matching_plan_records"):
            # Subplan matching is schema-level: F(p) is enough for S(p).  The
            # query template remains relevant for exact full-plan duplicates.
            return self.plan_repository.get_matching_plan_records(
                plan_fingerprint,
                None,
            )

        historical_latency, historical_plan = (
            self.plan_repository.get_execution_time_and_verbose_plan(
                plan_fingerprint,
                query_template,
            )
        )
        if historical_latency is None or historical_plan is None:
            return []
        return [(historical_latency, historical_plan)]

    def admission_check(
        self,
        sql_filepath: str,
        knobs: Dict[str, Union[int, float, str]],
        timeout_ms: float,
    ) -> Tuple[str, float | None, EvaluationStatus]:
        """
        Pre-check if plan exists and use subplan information to check if it will timeout.
        
        Returns:
            plan_fingerprint: candidate full-plan fingerprint.
            execution_time: Exact latency for duplicate plans, conservative
                estimate for subplan rejects, or None when the candidate is admitted.
            status: One of duplicate_plan, subplan_rejected, or admitted.
        """
        explain_result, _ = self._get_explain_result(sql_filepath, knobs)
        plan_fingerprint = self._get_plan_fingerprint(explain_result[0])
        query_template = self._repository_query_key(sql_filepath)
        historical_latency = self.plan_repository.get_execution_time(
            plan_fingerprint,
            query_template,
        )
        if historical_latency is not None:
            return plan_fingerprint, historical_latency, EvaluationStatus.DUPLICATE_PLAN

        subplan_matches = self._subplan_match(explain_result[0], query_template)
        if not subplan_matches:
            return plan_fingerprint, None, EvaluationStatus.ADMITTED

        max_time = 0
        for subplan_match in subplan_matches:
            subplan_time = estimate_subplan_execution_time(
                subplan_match["current_operator"],
                subplan_match["historical_plan"],
            )
            if subplan_time is not None:
                max_time = max(subplan_time, max_time)

        if max_time >= timeout_ms:
            logger.info("Subplan-based prediction will timeout")
            return plan_fingerprint, max_time, EvaluationStatus.SUBPLAN_REJECTED

        return plan_fingerprint, None, EvaluationStatus.ADMITTED

    def pre_check_with_subplan(
        self,
        sql_filepath: str,
        knobs: Dict[str, Union[int, float, str]],
        timeout_ms: float,
    ):
        """Backward-compatible admission check returning only plan and latency."""
        plan_fingerprint, execution_time, _ = self.admission_check(sql_filepath, knobs, timeout_ms)
        return plan_fingerprint, execution_time


    @property
    def timeout_threshold_ms(self) -> float | None:
        """Return the fixed Evaluator timeout threshold tau."""
        return self._timeout_threshold_ms

    def set_timeout_threshold(self, timeout_threshold_ms: float) -> float:
        """Set the fixed timeout threshold tau used by the Evaluator."""
        self._timeout_threshold_ms = float(timeout_threshold_ms)
        return self._timeout_threshold_ms

    def set_fail_time(self, fail_time: float) -> float:
        """Backward-compatible alias for older scripts."""
        return self.set_timeout_threshold(fail_time)

    @property
    def _fail_time(self) -> float | None:
        """Backward-compatible private alias for older tests/scripts."""
        return self._timeout_threshold_ms

    @_fail_time.setter
    def _fail_time(self, value: float | None) -> None:
        self._timeout_threshold_ms = None if value is None else float(value)

    @staticmethod
    def _is_statement_timeout_error(error: pymysql.MySQLError) -> bool:
        args = getattr(error, "args", ())
        message = str(error).lower()
        return (
            bool(args and args[0] == 3024)
            or "maximum statement execution time exceeded" in message
            or "query execution was interrupted" in message and "time" in message
        )
