# ruff: noqa: E402

import sys
import unittest
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from db.subplan import (
    estimate_subplan_execution_time,
    get_exec_time,
    parse_subplan_for_estimation,
    parse_subplan_time,
)

class TestSubplanParsing(unittest.TestCase):
    def test_get_exec_time_supports_common_tidb_units(self):
        self.assertEqual(get_exec_time({"executeInfo": "time:1.5s, loops:1"}), 1500.0)
        self.assertEqual(get_exec_time({"executeInfo": "time:250µs, loops:1"}), 0.25)
        self.assertEqual(get_exec_time({"executeInfo": "tikv_task:{proc max:2ms, min:0s}"}), 2.0)

    def test_parse_subplan_for_estimation_handles_no_tikv_node(self):
        plan_details = {
            "id": "Projection_1",
            "estRows": "1.00",
            "taskType": "root",
            "operatorInfo": "1",
        }
        self.assertEqual(parse_subplan_for_estimation(plan_details), (1.0, 0.0))

    def test_cop_tikv_subplan_estimate_uses_storage_component(self):
        cached_plan = {
            "id": "TableFullScan_1",
            "estRows": "100.0",
            "actRows": "100.0",
            "taskType": "cop[tikv]",
            "executeInfo": "tikv_task:{proc max:20ms, min:0s}",
            "operatorInfo": "keep order:false",
        }
        candidate_plan = {
            "id": "TableFullScan_2",
            "estRows": "250.0",
            "taskType": "cop[tikv]",
            "operatorInfo": "keep order:false",
        }

        self.assertEqual(parse_subplan_time(cached_plan), (0.0, 0, 20.0, 100.0))
        self.assertEqual(parse_subplan_for_estimation(candidate_plan), (0, 250.0))
        self.assertEqual(
            estimate_subplan_execution_time(candidate_plan, cached_plan),
            50.0,
        )

    def test_parse_subplan_time(self):
        plan_details = {
            "id": "TableReader_29(Build)",
            "estRows": "2609.13",
            "actRows": "1334883",
            "taskType": "root",
            "executeInfo": "time:1527.8602ms, open:63.1µs, close:4.67µs, loops:1308, cop_task: {num: 42, max: 848ms, min: 1.01ms, avg: 66.1ms, p95: 87.8ms, max_proc_keys: 1150127, p95_proc_keys: 50144, tot_proc: 2.44s, tot_wait: 5.61ms, copr_cache: disabled, build_task_duration: 36.2µs, max_distsql_concurrency: 3}, rpc_info:{Cop:{num_rpc:42, total_time:2.78s}}",
            "operatorInfo": "data:Selection_28, 84cf0a490cb12362c11bb03aa58784b5181fe781741164dbf9f78ba47f62c48a",
            "cost": "10392865435890.88",
            "memoryInfo": "1.53 MB",
            "diskInfo": "N/A",
            "subOperators": [
                {
                    "id": "Selection_28",
                    "estRows": "2609.13",
                    "actRows": "1334883",
                    "taskType": "cop[tikv]",
                    "executeInfo": "tikv_task:{proc max:847ms, min:0s, avg: 62.6ms, p80:71ms, p95:83ms, iters:2717, tasks:42}, scan_detail: {total_process_keys: 2609129, total_process_keys_size: 164576153, total_keys: 2609171, get_snapshot_time: 398.4µs, rocksdb: {key_skipped_count: 2609129, block: {cache_hit_count: 6440}}}, time_detail: {total_process_time: 2.44s, total_suspend_time: 255.1ms, total_wait_time: 5.61ms, total_kv_read_wall_time: 2.55s, tikv_grpc_process_time: 843.7µs, tikv_grpc_wait_time: 6.78ms, tikv_wall_time: 2.71s}",
                    "operatorInfo": "eq(imdb.movie_companies.company_type_id, 2), bb5443da55126ed8c5e76ee451088f27f1a424a50125f55258b6bcf2f4232cc0",
                    "cost": "132528889.18",
                    "memoryInfo": "N/A",
                    "diskInfo": "N/A",
                    "subOperators": [
                        {
                            "id": "TableFullScan_27",
                            "estRows": "2609129.00",
                            "actRows": "2609129",
                            "taskType": "cop[tikv]",
                            "accessObject": "table:mc",
                            "executeInfo": "tikv_task:{proc max:822ms, min:0s, avg: 60.7ms, p80:69ms, p95:81ms, iters:2717, tasks:42}",
                            "operatorInfo": "keep order:false, stats:partial[company_type_id:unInitialized], be78f849c99656fe3ceb08c56251ad222dbc3b73c248c6507ca8821405b876e9",
                            "cost": "2333352.08",
                            "memoryInfo": "N/A",
                            "diskInfo": "N/A"
                        }
                    ]
                }
            ]
        }
        result = parse_subplan_time(plan_details)
        self.assertEqual(result, (0.0, 0, 1527.8602, 2609129.0))

    def test_parse_subplan_time_2(self):
        plan_details = {
            "id": "HashJoin_27(Probe)",
            "estRows": "2610.24",
            "actRows": "1334882",
            "taskType": "root",
            "executeInfo": "time:2566.1944ms, open:77.1µs, close:14.1µs, loops:1308, build_hash_table:{concurrency:5, time:1.41s, fetch:1.34s, max_partition:59.4ms, total_partition:290.2ms, max_build:10.9ms, total_build:42ms}, probe:{concurrency:5, time:2.57s, fetch_and_wait:2.49s, max_worker_time:2.57s, total_worker_time:12.8s, max_probe:73.4ms, total_probe:349.6ms, probe_collision:214805}",
            "operatorInfo": "inner join, equal:[eq(imdb.title.id, imdb.movie_companies.movie_id)], b19801ff22056392a35d06fc8b6bc39e9cbbe8062b0ac93e7b500451cca9d0d1",
            "cost": "1183.47",
            "memoryInfo": "82.8 MB",
            "diskInfo": "0 Bytes",
            "subOperators": [
                {
                    "id": "TableReader_31(Build)",
                    "estRows": "2609.13",
                    "actRows": "1334883",
                    "taskType": "root",
                    "executeInfo": "time:1321.5484ms, open:27.5µs, close:2.82µs, loops:1307, cop_task: {num: 42, max: 649.2ms, min: 1.01ms, avg: 49.5ms, p95: 68.9ms, max_proc_keys: 1150127, p95_proc_keys: 50144, tot_proc: 1.95s, tot_wait: 1.48ms, copr_cache: disabled, build_task_duration: 11.5µs, max_distsql_concurrency: 3}, rpc_info:{Cop:{num_rpc:42, total_time:2.08s}}",
                    "operatorInfo": "data:Selection_30, 84cf0a490cb12362c11bb03aa58784b5181fe781741164dbf9f78ba47f62c48a",
                    "cost": "5.96",
                    "memoryInfo": "1.53 MB",
                    "diskInfo": "N/A",
                    "subOperators": [
                        {
                            "id": "Selection_30",
                            "estRows": "2609.13",
                            "actRows": "1334883",
                            "taskType": "cop[tikv]",
                            "executeInfo": "tikv_task:{proc max:646ms, min:0s, avg: 44.9ms, p80:56ms, p95:59ms, iters:2717, tasks:42}, scan_detail: {total_process_keys: 2609129, total_process_keys_size: 164576153, total_keys: 2609171, get_snapshot_time: 881.6µs, rocksdb: {key_skipped_count: 2609129, block: {cache_hit_count: 6440}}}, time_detail: {total_process_time: 1.95s, total_suspend_time: 3.54ms, total_wait_time: 1.48ms, total_kv_read_wall_time: 1.81s, tikv_grpc_process_time: 801.6µs, tikv_grpc_wait_time: 1.17ms, tikv_wall_time: 1.95s}",
                            "operatorInfo": "eq(imdb.movie_companies.company_type_id, 2), bb5443da55126ed8c5e76ee451088f27f1a424a50125f55258b6bcf2f4232cc0",
                            "cost": "130197039.20",
                            "memoryInfo": "N/A",
                            "diskInfo": "N/A",
                            "subOperators": [
                                {
                                    "id": "TableFullScan_29",
                                    "estRows": "2609129.00",
                                    "actRows": "2609129",
                                    "taskType": "cop[tikv]",
                                    "accessObject": "table:mc",
                                    "executeInfo": "tikv_task:{proc max:618ms, min:0s, avg: 43.1ms, p80:53ms, p95:57ms, iters:2717, tasks:42}",
                                    "operatorInfo": "keep order:false, stats:partial[company_type_id:unInitialized], be78f849c99656fe3ceb08c56251ad222dbc3b73c248c6507ca8821405b876e9",
                                    "cost": "1502.10",
                                    "memoryInfo": "N/A",
                                    "diskInfo": "N/A"
                                }
                            ]
                        }
                    ]
                },
                {
                    "id": "TableReader_33(Probe)",
                    "estRows": "2528311.00",
                    "actRows": "2528311",
                    "taskType": "root",
                    "executeInfo": "time:1060.1666ms, open:40.7µs, close:8.78µs, loops:2473, cop_task: {num: 59, max: 37.5ms, min: 578.5µs, avg: 19.7ms, p95: 24.6ms, max_proc_keys: 50144, p95_proc_keys: 50144, tot_proc: 1.08s, tot_wait: 2.24ms, copr_cache: disabled, build_task_duration: 14µs, max_distsql_concurrency: 2}, rpc_info:{Cop:{num_rpc:59, total_time:1.16s}}",
                    "operatorInfo": "data:TableFullScan_32, ccb0d7dbb386cbf97a7007105f4b9f48d79db972766c84bfba32e55df7ffede8",
                    "cost": "7.31",
                    "memoryInfo": "785.3 KB",
                    "diskInfo": "N/A",
                    "subOperators": [
                        {
                            "id": "TableFullScan_32",
                            "estRows": "2528311.00",
                            "actRows": "2528311",
                            "taskType": "cop[tikv]",
                            "accessObject": "table:t",
                            "executeInfo": "tikv_task:{proc max:22ms, min:0s, avg: 17.6ms, p80:21ms, p95:22ms, iters:2703, tasks:59}, scan_detail: {total_process_keys: 2528311, total_process_keys_size: 68264397, total_keys: 2528370, get_snapshot_time: 1.06ms, rocksdb: {key_skipped_count: 2528311, block: {cache_hit_count: 9913}}}, time_detail: {total_process_time: 1.08s, total_suspend_time: 1.09ms, total_wait_time: 2.24ms, total_kv_read_wall_time: 1.04s, tikv_grpc_process_time: 751.5µs, tikv_grpc_wait_time: 1.15ms, tikv_wall_time: 1.09s}",
                            "operatorInfo": "keep order:false, 15b85fa9285062ff5180a900e59d9fda09ed7e8a3bfb3a5f00ec5fb3e6855cf4",
                            "cost": "1513.24",
                            "memoryInfo": "N/A",
                            "diskInfo": "N/A"
                        }
                    ]
                }
            ]
        }
        result = parse_subplan_time(plan_details)
        self.assertEqual(result, (1244.646, 1334882.0, 1321.5484, 2609129.0))

    def test_parse_subplan_for_estimation(self):
        plan_details = {
            "id": "HashAgg_14",
            "estRows": "1.00",
            "taskType": "root",
            "operatorInfo": "funcs:count(1)->Column#23, 49a166d595274027247b685fc6b78e5bceb403cd09d6bd708d8a62171d5ab63c",
            "cost": "1505.98",
            "subOperators": [
                {
                    "id": "HashJoin_17",
                    "estRows": "1725.78",
                    "taskType": "root",
                    "operatorInfo": "inner join, equal:[eq(imdb.title.id, imdb.movie_companies.movie_id)], c2b26ef4a8050c42ca0b46c1f6a3c1fce3779947b743819cbf65edab3f48da28",
                    "cost": "1957117389.46",
                    "subOperators": [
                        {
                            "id": "HashJoin_27(Build)",
                            "estRows": "1380.62",
                            "taskType": "root",
                            "operatorInfo": "inner join, equal:[eq(imdb.title.id, imdb.movie_info_idx.movie_id)], 4e28f949b2c9a2dc35e679bb8e98812d4343907b83dad10c9cb04e984831a280",
                            "cost": "3107829632.71",
                            "subOperators": [
                                {
                                    "id": "TableReader_31(Build)",
                                    "estRows": "1380.04",
                                    "taskType": "root",
                                    "operatorInfo": "data:Selection_30, 657be9251b6f282a0f8cdca66b7d65f666a44bb64f3e9712cdb27b96264ae953",
                                    "cost": "87871895568843.19",
                                    "subOperators": [
                                        {
                                            "id": "Selection_30",
                                            "estRows": "1380.04",
                                            "taskType": "cop[tikv]",
                                            "operatorInfo": "eq(imdb.movie_info_idx.info_type_id, 112), 35b09efc82228842fff989a6346a254eb8cf68cf1d4717b13c235782490f4c2f",
                                            "cost": "4166191287139.07",
                                            "subOperators": [
                                                {
                                                    "id": "TableFullScan_29",
                                                    "estRows": "1380035.00",
                                                    "taskType": "cop[tikv]",
                                                    "accessObject": "table:mi_idx",
                                                    "operatorInfo": "keep order:false, stats:partial[info_type_id:unInitialized], 2402544ff2d4013ca38602a4c0c1ca6e97166ddc4afd73e5339aa6367dd0cbc0",
                                                    "cost": "4166122423392.57"
                                                }
                                            ]
                                        }
                                    ]
                                },
                                {
                                    "id": "TableReader_33(Probe)",
                                    "estRows": "2528311.00",
                                    "taskType": "root",
                                    "operatorInfo": "data:TableFullScan_32, ccb0d7dbb386cbf97a7007105f4b9f48d79db972766c84bfba32e55df7ffede8",
                                    "cost": "152476328022568.41",
                                    "subOperators": [
                                        {
                                            "id": "TableFullScan_32",
                                            "estRows": "2528311.00",
                                            "taskType": "cop[tikv]",
                                            "accessObject": "table:t",
                                            "operatorInfo": "keep order:false, 15b85fa9285062ff5180a900e59d9fda09ed7e8a3bfb3a5f00ec5fb3e6855cf4",
                                            "cost": "7229063346404.42"
                                        }
                                    ]
                                }
                            ]
                        },
                        {
                            "id": "TableReader_36(Probe)",
                            "estRows": "2609.13",
                            "taskType": "root",
                            "operatorInfo": "data:Selection_35, 84cf0a490cb12362c11bb03aa58784b5181fe781741164dbf9f78ba47f62c48a",
                            "cost": "151353253993274.28",
                            "subOperators": [
                                {
                                    "id": "Selection_35",
                                    "estRows": "2609.13",
                                    "taskType": "cop[tikv]",
                                    "operatorInfo": "eq(imdb.movie_companies.company_type_id, 2), bb5443da55126ed8c5e76ee451088f27f1a424a50125f55258b6bcf2f4232cc0",
                                    "cost": "7175975906748.11",
                                    "subOperators": [
                                        {
                                            "id": "TableFullScan_34",
                                            "estRows": "2609129.00",
                                            "taskType": "cop[tikv]",
                                            "accessObject": "table:mc",
                                            "operatorInfo": "keep order:false, stats:partial[company_type_id:unInitialized], be78f849c99656fe3ceb08c56251ad222dbc3b73c248c6507ca8821405b876e9",
                                            "cost": "7175845711211.01"
                                        }
                                    ]
                                }
                            ]
                        }
                    ]
                }
            ]
        }
        result = parse_subplan_for_estimation(plan_details)
        self.assertEqual(result, (3107.3999999999996, 2609129.0))

if __name__ == "__main__":
    unittest.main()  
