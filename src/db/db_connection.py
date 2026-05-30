from contextlib import suppress

import pymysql
from pymysql.connections import Connection as MySQLConnection
from pymysql.cursors import DictCursor

from util.config import AppConfig, load_app_config
from util.logger import logger

class DBConnectionFactory:
    """Factory for creating MySQL database connections."""

    @staticmethod
    def create_connection(
        autocommit: bool | None = None,
        app_config: AppConfig | None = None,
    ) -> MySQLConnection:
        """Create and return a MySQL connection with optional autocommit."""
        config = app_config or load_app_config()
        database = config.database
        use_autocommit = database.autocommit if autocommit is None else autocommit
        connection = pymysql.connect(
            host=database.host,
            port=database.port,
            user=database.user,
            password=database.password,
            database=database.name,
            autocommit=use_autocommit,
            cursorclass=DictCursor,
            ssl={'ca': database.ca_path} if database.ca_path else None
        )
        DBConnectionFactory._initialize_session_settings(
            connection,
            mem_quota_bytes=database.mem_quota_bytes,
            validate_copr_cache=database.validate_copr_cache,
        )
        return connection

    @staticmethod
    def _initialize_session_settings(
        connection: MySQLConnection,
        mem_quota_bytes: int,
        validate_copr_cache: bool = True,
    ) -> None:
        """Initialize session variables for TiDB connection."""
        with connection.cursor() as cursor:
            try:
                cursor.execute(f"SET SESSION tidb_mem_quota_query = {mem_quota_bytes};")
                cursor.execute("SELECT @@tidb_mem_quota_query;")
                mem_quota = cursor.fetchone()
                logger.info(f"tidb_mem_quota_query: {mem_quota}")
            except pymysql.MySQLError as error:
                if getattr(error, "args", [None])[0] != 1193:
                    raise
                logger.warning("tidb_mem_quota_query is unsupported; skipping session memory quota")

            if not validate_copr_cache:
                return

            try:
                cursor.execute("show config where name='tikv-client.copr-cache.capacity-mb';")
                copr_cache = cursor.fetchone()
            except pymysql.MySQLError as error:
                logger.warning(f"Failed to validate copr-cache setting: {error}")
                return
            logger.info(f"copr-cache: {copr_cache}")

            cache_raw = None
            if copr_cache:
                cache_raw = copr_cache.get("Value")
                if cache_raw is None:
                    cache_raw = copr_cache.get("value")
            if cache_raw is None:
                logger.warning("Failed to validate copr-cache setting: missing value")
                return

            try:
                cache_value = float(str(cache_raw))
            except (TypeError, ValueError) as error:
                logger.warning(f"Failed to parse copr-cache setting: {error}")
                return

            if cache_value != 0.0:
                logger.error(
                    "tikv copr-cache must be disabled for fair benchmarking. "
                    "Current capacity-mb=%s. Set "
                    "tikv-client.copr-cache.capacity-mb=0 and retry.",
                    cache_value,
                )
                with suppress(Exception):
                    connection.close()
                raise RuntimeError(
                    "tikv copr-cache must be disabled for fair benchmarking "
                    f"(current capacity-mb={cache_value})"
                )
