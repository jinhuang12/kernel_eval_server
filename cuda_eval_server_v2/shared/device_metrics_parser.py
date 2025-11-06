"""
Device metrics parser using NCU report interface
Extracts Speed of Light and other performance metrics from NCU profiling reports
"""

import logging
import os
import sys
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import ncu_report from NCU installation
NCU_REPORT_AVAILABLE = False
try:
    # First try to find NCU installation and add ncu_report to path
    ncu_paths = [
        "/opt/nvidia/nsight-compute/*/extras/python",
        "/usr/local/cuda/nsight-compute*/extras/python",
        "/usr/local/cuda/bin/"
    ]
    
    ncu_report_path = None
    for path_pattern in ncu_paths:
        import glob
        matches = glob.glob(path_pattern)
        if matches:
            ncu_report_path = matches[-1]  # Use last match
            break
    
    if ncu_report_path:
        sys.path.insert(0, ncu_report_path)
        import ncu_report
        NCU_REPORT_AVAILABLE = True
        logger.info(f"NCU report interface loaded from: {ncu_report_path}")
    else:
        logger.warning("NCU installation not found - device metrics will be unavailable")
        
except ImportError as e:
    logger.warning(f"Could not import ncu_report: {e} - device metrics will be unavailable")

# -------------------------------
# Helpers
# -------------------------------

def _metric_value(action, name: str) -> Tuple[Optional[float], Optional[str]]:
    """Return (value, unit) if metric exists and is numeric, else (None, None)."""
    try:
        if name in action:
            m = action[name]
            v = m.value()
            if isinstance(v, (int, float)):
                try:
                    unit = m.unit()  # may raise if units not exposed
                except Exception:
                    unit = None
                return float(v), unit
    except Exception as e:
        logger.debug(f"Could not extract metric {name}: {e}")
    return None, None


def _to_seconds(val: float, unit: Optional[str]) -> Optional[float]:
    """Convert a duration to seconds if unit is known; otherwise return None."""
    if val is None:
        return None
    if not unit:
        return None
    u = unit.lower()
    if u in ("s", "sec", "second", "seconds"):
        return float(val)
    if u in ("ms", "millisecond", "milliseconds"):
        return float(val) / 1e3
    if u in ("us", "µs", "microsecond", "microseconds"):
        return float(val) / 1e6
    if u in ("ns", "nanosecond", "nanoseconds"):
        return float(val) / 1e9
    # cycles and other units need device clocks; we skip conversion here
    return None


def _safe_div(n: Optional[float], d: Optional[float]) -> Optional[float]:
    try:
        if n is None or d is None or d == 0:
            return None
        return float(n) / float(d)
    except Exception:
        return None


class DeviceMetricsParser:
    """Parser for NCU report files to extract device performance metrics"""
    
    def __init__(self, report_file_path: Optional[str] = None):
        self.report_file_path = report_file_path
        self.available = NCU_REPORT_AVAILABLE and report_file_path # and os.path.exists(report_file_path)
        
        if not NCU_REPORT_AVAILABLE:
            logger.warning("NCU report interface not available - no device metrics will be collected")
        elif report_file_path and not os.path.exists(report_file_path):
            logger.warning(f"NCU report file not found: {report_file_path}")
    
    def is_available(self) -> bool:
        """Check if device metrics parsing is available"""
        return self.available
    
    def update_report_file(self, report_file_path: str):
        """Update the report file path and check availability"""
        self.report_file_path = report_file_path
        self.available = NCU_REPORT_AVAILABLE and os.path.exists(report_file_path)
        
        if self.available:
            logger.info(f"Updated NCU report file: {report_file_path}")
        else:
            logger.warning(f"NCU report file not available: {report_file_path}")
    
    def get_metrics_for_request(self, request_id: str) -> Dict[str, Any]:
        """
        Parse NCU report and extract metrics for specific request
        
        Args:
            request_id: The request ID used in NVTX range names
            
        Returns:
            Dictionary with original_device_metrics and custom_device_metrics
        """
        if not self.available:
            logger.debug(f"Device metrics not available for request {request_id}")
            return {}
        
        try:
            logger.info(f"Parsing device metrics for request {request_id} from {self.report_file_path}")
            
            # Load the NCU report
            report = ncu_report.load_report(self.report_file_path)
            
            # Get metrics for both original and custom execution
            original_metrics = self._extract_range_metrics(report, f"{request_id}_original")
            custom_metrics = self._extract_range_metrics(report, f"{request_id}_custom")
            
            result = {}
            if original_metrics:
                result["original_device_metrics"] = original_metrics
            if custom_metrics:
                result["custom_device_metrics"] = custom_metrics
            
            if result:
                logger.info(f"Successfully extracted device metrics for request {request_id}")
            else:
                logger.warning(f"No device metrics found for request {request_id}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to parse device metrics for request {request_id}: {e}")
            return {}
    
    def _extract_range_metrics(self, report, nvtx_range: str) -> Dict[str, Any]:
        """
        Extract Speed of Light and other key metrics from specific NVTX range
        
        Args:
            report: Loaded NCU report
            nvtx_range: NVTX range name to filter by
            
        Returns:
            Dictionary with extracted metrics
        """
        metrics = {}
        found_actions = 0
        
        try:
            # Iterate through all ranges in the report
            for range_idx in range(report.num_ranges()):
                current_range = report.range_by_idx(range_idx)
                
                # Filter actions by NVTX range using push/pop syntax
                matching_actions = current_range.actions_by_nvtx([f"{nvtx_range}/"], [])
                
                if not matching_actions:
                    continue
                
                logger.debug(f"Found {len(matching_actions)} actions for NVTX range: {nvtx_range}")
                
                # Process all matching actions (typically kernels)
                for action_idx in matching_actions:
                    action = current_range.action_by_idx(action_idx)
                    found_actions += 1
                    
                    logger.debug(f"Processing action: {action.name()}")
                    
                    # Extract Speed of Light metrics (primary focus)
                    speed_of_light = self._extract_speed_of_light_metrics(action)
                    if speed_of_light:
                        metrics["speed_of_light"] = speed_of_light
                    
                    # Extract additional key performance metrics
                    detailed_metrics = self._extract_detailed_metrics(action)
                    if detailed_metrics:
                        metrics["detailed_metrics"] = detailed_metrics
                    
                    # Extract memory hierarchy metrics
                    memory_metrics = self._extract_memory_metrics(action)
                    if memory_metrics:
                        metrics["memory_metrics"] = memory_metrics
                    
                    # Extract compute utilization metrics  
                    compute_metrics = self._extract_compute_metrics(action)
                    if compute_metrics:
                        metrics["compute_metrics"] = compute_metrics
                    
                    # Extract pipeline utilization metrics
                    pipeline_metrics = self._extract_pipeline_metrics(action)
                    if pipeline_metrics:
                        metrics["pipeline_metrics"] = pipeline_metrics
                    
                    # Extract occupancy-specific metrics
                    occupancy_metrics = self._extract_occupancy_metrics(action)
                    if occupancy_metrics:
                        metrics["occupancy_metrics"] = occupancy_metrics

                    # NEW: Stall reasons (issue-side)
                    stall_metrics = self._extract_stall_metrics(action)
                    if stall_metrics:
                        metrics["stall_metrics"] = stall_metrics

                    # NEW: Scheduler signals (eligible vs issued, etc.)
                    sched_metrics = self._extract_scheduler_metrics(action)
                    if sched_metrics:
                        metrics["scheduler_metrics"] = sched_metrics

                    # NEW: Access-pattern diagnostics (coalescing, L2 over-fetch, bank conflicts)
                    access_metrics = self._extract_access_pattern_metrics(action)
                    if access_metrics:
                        metrics["access_pattern_metrics"] = access_metrics

                    # NEW: Roofline (AI, FLOP/s) where available
                    roofline_metrics = self._extract_roofline_metrics(action, memory_metrics)
                    if roofline_metrics:
                        metrics["roofline_metrics"] = roofline_metrics

                    # NEW: Timing/cycle context
                    timing_metrics = self._extract_timing_metrics(action)
                    if timing_metrics:
                        metrics["timing_metrics"] = timing_metrics    
        
        except Exception as e:
            logger.error(f"Error extracting metrics for range {nvtx_range}: {e}")
        
        if found_actions == 0:
            logger.warning(f"No actions found for NVTX range: {nvtx_range}")
        else:
            logger.debug(f"Processed {found_actions} actions for range: {nvtx_range}")
        
        return metrics
    
    def _extract_speed_of_light_metrics(self, action) -> Dict[str, float]:
        """Extract Speed of Light throughput metrics"""
        speed_of_light = {}
        
        # Speed of Light metric names (may vary by GPU architecture)
        sol_metrics = [
            # Compute throughput
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
            
            # Memory throughput  
            "dram__throughput.avg.pct_of_peak_sustained_elapsed",
            "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
        ]
        
        for metric_name in sol_metrics:
            try:
                if metric_name in action:
                    metric = action[metric_name]
                    value = metric.value()
                    if isinstance(value, (int, float)) and value is not None:
                        # Map to simplified names expected by the validator
                        if "gpu__compute_memory_throughput" in metric_name:
                            speed_of_light["compute_memory_throughput_pct"] = float(value)
                            # Also set as primary compute throughput for validator
                            speed_of_light["compute_throughput_pct"] = float(value)
                        elif "sm__throughput" in metric_name:
                            speed_of_light["sm_throughput_pct"] = float(value)
                            # Use SM throughput as compute if not already set
                            if "compute_throughput_pct" not in speed_of_light:
                                speed_of_light["compute_throughput_pct"] = float(value)
                        elif "gpu__dram_throughput" in metric_name:
                            speed_of_light["gpu_dram_throughput_pct"] = float(value)
                            # Also set as primary memory throughput for validator
                            speed_of_light["memory_throughput_pct"] = float(value)
                        elif "dram__throughput" in metric_name:
                            speed_of_light["dram_throughput_pct"] = float(value)
                            # Use DRAM throughput as memory if not already set
                            if "memory_throughput_pct" not in speed_of_light:
                                speed_of_light["memory_throughput_pct"] = float(value)
            except Exception as e:
                logger.debug(f"Could not extract metric {metric_name}: {e}")
        
        return speed_of_light
    
    def _extract_detailed_metrics(self, action) -> Dict[str, float]:
        """Extract detailed performance metrics beyond Speed of Light"""
        detailed = {}
        
        # Key performance indicators for bottleneck analysis
        detailed_metrics = [
            "l1tex__t_sector_hit_rate.pct",  # L1 cache hit rate
            "lts__t_sector_hit_rate.pct",    # L2 cache hit rate  
            "sm__warps_active.avg.pct_of_peak_sustained_active",  # Warp occupancy
            "sm__cycles_active.avg.pct_of_peak_sustained_elapsed", # SM active cycles
            "sm__inst_executed.avg.per_cycle_active",  # Instructions per cycle
            "launch__waves_per_multiprocessor",  # Waves per SM
        ]
        
        metric_name_mapping = {
            "l1tex__t_sector_hit_rate.pct": "l1_hit_rate_pct",
            "lts__t_sector_hit_rate.pct": "l2_hit_rate_pct", 
            "sm__warps_active.avg.pct_of_peak_sustained_active": "warp_occupancy_pct",
            "sm__cycles_active.avg.pct_of_peak_sustained_elapsed": "sm_active_cycles_pct",
            "sm__inst_executed.avg.per_cycle_active": "instructions_per_cycle",
            "launch__waves_per_multiprocessor": "waves_per_sm",
        }
        
        for metric_name in detailed_metrics:
            try:
                if metric_name in action:
                    metric = action[metric_name]
                    value = metric.value()
                    if isinstance(value, (int, float)) and value is not None:
                        simple_name = metric_name_mapping.get(metric_name, metric_name)
                        detailed[simple_name] = float(value)
            except Exception as e:
                logger.debug(f"Could not extract detailed metric {metric_name}: {e}")
        
        return detailed
    
    def _extract_memory_metrics(self, action) -> Dict[str, float]:
        """Extract memory-related performance metrics"""
        memory = {}
        
        # Use combined dram__bytes metrics and L1 metrics that actually exist
        memory_metrics = [
            "dram__bytes.avg.per_second",    # Combined DRAM bandwidth
            "dram__bytes.sum.per_second",    # Total DRAM bandwidth
            "dram__cycles_active.avg.pct_of_peak_sustained_elapsed",  # DRAM active cycles
            "l1tex__lsu_writeback_active.avg.pct_of_peak_sustained_elapsed", # L1 writeback
            "l1tex__m_xbar2l1tex_read_sectors.avg.pct_of_peak_sustained_elapsed", # L1 read sectors
            "lts__throughput.avg.pct_of_peak_sustained_elapsed",  # L2 throughput
        ]
        
        metric_name_mapping = {
            "dram__bytes.avg.per_second": "dram_avg_bandwidth_gb_s",
            "dram__bytes.sum.per_second": "dram_total_bandwidth_gb_s",
            "dram__cycles_active.avg.pct_of_peak_sustained_elapsed": "dram_active_cycles_pct",
            "l1tex__lsu_writeback_active.avg.pct_of_peak_sustained_elapsed": "l1_writeback_active_pct",
            "l1tex__m_xbar2l1tex_read_sectors.avg.pct_of_peak_sustained_elapsed": "l1_read_sectors_pct",
            "lts__throughput.avg.pct_of_peak_sustained_elapsed": "l2_throughput_pct",
        }
        
        for metric_name in memory_metrics:
            try:
                if metric_name in action:
                    metric = action[metric_name]  
                    value = metric.value()
                    if isinstance(value, (int, float)) and value is not None:
                        simple_name = metric_name_mapping.get(metric_name, metric_name)
                        # Convert bytes/second to GB/s for bandwidth metrics
                        if "bandwidth" in simple_name and "bytes" in metric_name:
                            memory[simple_name] = float(value) / (1024**3)
                        else:
                            memory[simple_name] = float(value)
            except Exception as e:
                logger.debug(f"Could not extract memory metric {metric_name}: {e}")
        
        return memory
    
    def _extract_compute_metrics(self, action) -> Dict[str, float]:
        """Extract compute-related performance metrics"""
        compute = {}
        
        # Use pipeline-specific metrics that actually exist in NCU
        compute_metrics = [
            "sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active", # FMA pipe utilization
            "sm__inst_executed_pipe_fp64.avg.pct_of_peak_sustained_active", # FP64 pipe utilization
            "sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active", # ALU pipe utilization
            "sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active", # XU pipe utilization
            "sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active", # Tensor ops
            "sm__inst_executed.avg.per_cycle_active", # IPC
            "launch__occupancy_limit_blocks",          # Occupancy limiting factor (blocks)
            "launch__occupancy_limit_registers",       # Occupancy limiting factor (registers)
            "launch__occupancy_limit_shared_mem",      # Occupancy limiting factor (shared mem)
            "launch__occupancy_limit_warps",           # Occupancy limiting factor (warps)
            "launch__registers_per_thread",           # Register pressure
        ]
        
        metric_name_mapping = {
            "sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active": "fma_pipe_utilization_pct",
            "sm__inst_executed_pipe_fp64.avg.pct_of_peak_sustained_active": "fp64_pipe_utilization_pct",
            "sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active": "alu_pipe_utilization_pct",
            "sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active": "xu_pipe_utilization_pct",
            "sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active": "tensor_pipe_utilization_pct",
            "sm__inst_executed.avg.per_cycle_active": "instructions_per_cycle",
            "launch__occupancy_limit_blocks": "occupancy_limit_blocks",
            "launch__occupancy_limit_registers": "occupancy_limit_registers",
            "launch__occupancy_limit_shared_mem": "occupancy_limit_shared_mem",
            "launch__occupancy_limit_warps": "occupancy_limit_warps",
            "launch__registers_per_thread": "registers_per_thread",
        }
        
        for metric_name in compute_metrics:
            try:
                if metric_name in action:
                    metric = action[metric_name]
                    value = metric.value()
                    if isinstance(value, (int, float)) and value is not None:
                        simple_name = metric_name_mapping.get(metric_name, metric_name)
                        compute[simple_name] = float(value)
            except Exception as e:
                logger.debug(f"Could not extract compute metric {metric_name}: {e}")
        
        return compute
    
    def _extract_pipeline_metrics(self, action) -> Dict[str, float]:
        """Extract compute pipeline utilization metrics"""
        pipeline = {}
        
        pipeline_metrics = [
            "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed",
            "sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_elapsed",
            "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
            "sm__pipe_shared_cycles_active.avg.pct_of_peak_sustained_elapsed",
            "sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_elapsed",
            "sm__issue_active.avg.pct_of_peak_sustained_elapsed",
        ]
        
        metric_name_mapping = {
            "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed": "fma_pipe_active_pct",
            "sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_elapsed": "alu_pipe_active_pct",
            "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed": "tensor_pipe_active_pct",
            "sm__pipe_shared_cycles_active.avg.pct_of_peak_sustained_elapsed": "shared_pipe_active_pct",
            "sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_elapsed": "fp64_pipe_active_pct",
            "sm__issue_active.avg.pct_of_peak_sustained_elapsed": "sm_issue_active_pct",
        }
        
        for metric_name in pipeline_metrics:
            try:
                if metric_name in action:
                    metric = action[metric_name]
                    value = metric.value()
                    if isinstance(value, (int, float)) and value is not None:
                        simple_name = metric_name_mapping.get(metric_name, metric_name)
                        pipeline[simple_name] = float(value)
            except Exception as e:
                logger.debug(f"Could not extract pipeline metric {metric_name}: {e}")
        
        return pipeline
    
    def _extract_occupancy_metrics(self, action) -> Dict[str, float]:
        """Extract occupancy-related metrics"""
        occupancy = {}
        
        occupancy_metrics = [
            "launch__occupancy_limit_registers",
            "launch__occupancy_limit_shared_mem",
            "launch__occupancy_limit_warps",
            "launch__occupancy_limit_blocks",
            "launch__waves_per_multiprocessor",
            "launch__block_size",
            "launch__grid_size",
            "launch__shared_mem_per_block",
        ]
        
        metric_name_mapping = {
            "launch__occupancy_limit_registers": "occupancy_limit_registers",
            "launch__occupancy_limit_shared_mem": "occupancy_limit_shared_mem",
            "launch__occupancy_limit_warps": "occupancy_limit_warps",
            "launch__occupancy_limit_blocks": "occupancy_limit_blocks",
            "launch__waves_per_multiprocessor": "waves_per_sm",
            "launch__block_size": "block_size",
            "launch__grid_size": "grid_size",
            "launch__shared_mem_per_block": "shared_mem_per_block",
        }
        
        for metric_name in occupancy_metrics:
            try:
                if metric_name in action:
                    metric = action[metric_name]
                    value = metric.value()
                    if isinstance(value, (int, float)) and value is not None:
                        simple_name = metric_name_mapping.get(metric_name, metric_name)
                        occupancy[simple_name] = float(value)
            except Exception as e:
                logger.debug(f"Could not extract occupancy metric {metric_name}: {e}")
        
        return occupancy
    
        # -------------------------------
    # New extractors (added)
    # -------------------------------

    def _extract_stall_metrics(self, action) -> Dict[str, float]:
        """Extract warp stall reasons (issue-side)."""
        stalls: Dict[str, float] = {}
        pairs = {
            "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct": "stall_long_scoreboard_pct",
            "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct": "stall_short_scoreboard_pct",
            "smsp__warp_issue_stalled_barrier_per_warp_active.pct": "stall_barrier_pct",
            "smsp__warp_issue_stalled_not_selected_per_warp_active.pct": "stall_not_selected_pct",
        }
        for m, k in pairs.items():
            v, _ = _metric_value(action, m)
            if v is not None:
                stalls[k] = v
        return stalls

    def _extract_scheduler_metrics(self, action) -> Dict[str, float]:
        """Extract scheduler eligibility/issue stats when available."""
        sched: Dict[str, float] = {}
        pairs = {
            "smsp__warps_eligible.avg.per_cycle_active": "warps_eligible_per_cycle",
            "smsp__inst_issued.avg.per_cycle_active": "inst_issued_per_cycle",
            "smsp__issue_active.avg.pct_of_peak_sustained_active": "issue_active_pct",
        }
        for m, k in pairs.items():
            v, _ = _metric_value(action, m)
            if v is not None:
                sched[k] = v
        return sched

    def _extract_access_pattern_metrics(self, action) -> Dict[str, float]:
        """Access‑pattern diagnostics: coalescing, L2 over‑fetch, bank conflicts."""
        ap: Dict[str, float] = {}

        # L1 requests vs sectors (loads/stores) → coalescing score
        req_ld, _ = _metric_value(action, "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum")
        sec_ld, _ = _metric_value(action, "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum")
        req_st, _ = _metric_value(action, "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum")
        sec_st, _ = _metric_value(action, "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum")

        ldr = _safe_div(sec_ld, req_ld)
        strr = _safe_div(sec_st, req_st)
        if ldr is not None:
            ap["l1_load_sectors_per_req"] = ldr
        if strr is not None:
            ap["l1_store_sectors_per_req"] = strr

        # L2 theoretical sectors (ideal/excessive)
        l2_all, _   = _metric_value(action, "memory_l2_theoretical_sectors_global")
        l2_ideal, _ = _metric_value(action, "memory_l2_theoretical_sectors_global_ideal")
        # Try derived metric (may exist in some reports), otherwise compute difference
        l2_excess, _ = _metric_value(action, "derived__memory_l2_theoretical_sectors_global_excessive")

        if l2_all is not None:
            ap["l2_theoretical_sectors_global"] = l2_all
        if l2_ideal is not None:
            ap["l2_theoretical_sectors_global_ideal"] = l2_ideal
        if l2_excess is None and (l2_all is not None and l2_ideal is not None):
            l2_excess = max(l2_all - l2_ideal, 0.0)
        if l2_excess is not None:
            ap["l2_theoretical_sectors_global_excessive"] = l2_excess

        l2_excess_frac = _safe_div(l2_excess, l2_all)
        if l2_excess_frac is not None:
            ap["l2_excess_frac"] = l2_excess_frac

        # Shared memory bank conflicts
        bank_ld, _ = _metric_value(action, "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum")
        bank_st, _ = _metric_value(action, "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum")
        if bank_ld is not None:
            ap["shared_bank_conflicts_load_sum"] = bank_ld
        if bank_st is not None:
            ap["shared_bank_conflicts_store_sum"] = bank_st

        return ap


    def _extract_roofline_metrics(self, action, memory_metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Roofline quantities: arithmetic intensity (FLOPs / DRAM bytes) and GFLOP/s if time is known.
        We try common derived names from InstructionStats / Roofline; fall back gracefully.
        """
        roof: Dict[str, float] = {}

        # Try to gather FLOP counts (sum of available kinds)
        flop_names = [
            "flop_count_sp",
            "flop_count_hp",
            "flop_count_dp",
            "flop_count_tensor",
        ]
        total_flops = 0.0
        found_any_flops = False
        for fn in flop_names:
            v, _ = _metric_value(action, fn)
            if v is not None:
                total_flops += float(v)
                found_any_flops = True

        if found_any_flops:
            roof["flop_count_total"] = total_flops

        # DRAM bytes moved (prefer direct sum; else reconstruct)
        dram_sum_bytes, u_bytes = _metric_value(action, "dram__bytes.sum")
        if dram_sum_bytes is None:
            # reconstruct from avg per second * duration if both exist and duration unit known
            dram_avg_Bps_raw, _ = _metric_value(action, "dram__bytes.avg.per_second")
            dur_raw, dur_unit = _metric_value(action, "gpu__time_duration.sum")
            dur_s = _to_seconds(dur_raw, dur_unit) if dur_raw is not None else None
            if dram_avg_Bps_raw is not None and dur_s is not None:
                dram_sum_bytes = float(dram_avg_Bps_raw) * float(dur_s)

        if dram_sum_bytes is not None:
            roof["dram_bytes_sum"] = float(dram_sum_bytes)

        # Arithmetic intensity
        if found_any_flops and dram_sum_bytes is not None and dram_sum_bytes > 0:
            roof["arithmetic_intensity"] = float(total_flops) / float(dram_sum_bytes)

        # GFLOP/s if we have time in seconds
        dur_raw, dur_unit = _metric_value(action, "gpu__time_duration.sum")
        dur_s = _to_seconds(dur_raw, dur_unit) if dur_raw is not None else None
        if found_any_flops and dur_s is not None and dur_s > 0:
            roof["gflops"] = (total_flops / dur_s) / 1e9

        return roof

    def _extract_timing_metrics(self, action) -> Dict[str, float]:
        """Collect duration and cycles to enable normalization externally."""
        tm: Dict[str, float] = {}
        dur_val, dur_unit = _metric_value(action, "gpu__time_duration.sum")
        if dur_val is not None:
            tm["gpu_time_duration_sum"] = dur_val
            if dur_unit:
                tm["gpu_time_duration_unit"] = dur_unit  # keep raw unit for later normalization
            # Add seconds if convertible
            dur_s = _to_seconds(dur_val, dur_unit)
            if dur_s is not None:
                tm["gpu_time_duration_seconds"] = dur_s

        cycles_val, cycles_unit = _metric_value(action, "gpc__cycles_elapsed.max")
        if cycles_val is not None:
            tm["gpc_cycles_elapsed_max"] = cycles_val
            if cycles_unit:
                tm["gpc_cycles_elapsed_unit"] = cycles_unit

        return tm

    # -------------------------------
    
    def get_available_metrics(self) -> List[str]:
        """
        Get list of all available metrics in the report (for debugging)
        
        Returns:
            List of metric names found in the report
        """
        if not self.available:
            return []
        
        try:
            report = ncu_report.load_report(self.report_file_path)
            all_metrics = set()
            
            for range_idx in range(report.num_ranges()):
                current_range = report.range_by_idx(range_idx)
                
                for action_idx in range(current_range.num_actions()):
                    action = current_range.action_by_idx(action_idx)
                    all_metrics.update(action.metric_names())
            
            return sorted(list(all_metrics))
            
        except Exception as e:
            logger.error(f"Could not get available metrics: {e}")
            return []


def create_global_metrics_parser(report_file_path: Optional[str] = None) -> DeviceMetricsParser:
    """Create a global device metrics parser instance"""
    return DeviceMetricsParser(report_file_path)


# Global metrics parser instance (will be initialized when NCU report is available)
_global_metrics_parser: Optional[DeviceMetricsParser] = None


def get_device_metrics_parser() -> Optional[DeviceMetricsParser]:
    """Get the global device metrics parser instance"""
    return _global_metrics_parser


def set_device_metrics_parser(parser: DeviceMetricsParser):
    """Set the global device metrics parser instance"""
    global _global_metrics_parser
    _global_metrics_parser = parser
