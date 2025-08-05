"""
Main GPUWarmTier class.
This class acts as a coordinator, delegating tasks to specialized managers.
"""
from typing import List, AsyncIterator, Optional, Dict
from pathlib import Path
import pyarrow as pa
from .interfaces import StorageTier, EvictionCallback
from .wal_manager import WALManager
from .logger import get_logger

from .warm_tier_cache import GPUCacheManager
from .warm_tier_query import GPUQueryHandler
from .warm_tier_eviction import GPUEvictionManager

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None

class GPUWarmTier(StorageTier):
    """
    GPU-only storage tier that coordinates caching, querying, and eviction.
    """
    
    def __init__(self, max_memory_mb: int = 2048, wal_dir: str = "./wal", debug: bool = False, config=None):
        # Use config values if available, otherwise use parameters
        if config is not None:
            self.max_memory_mb = config.warm_tier.max_memory_mb
            self.eviction_threshold_pct = config.warm_tier.eviction_threshold_pct
            self.wal_enabled = config.warm_tier.wal_enabled
        else:
            self.max_memory_mb = max_memory_mb
            self.eviction_threshold_pct = 0.8  # Default fallback
            self.wal_enabled = True
        
        # GPU is enabled if WAL is enabled (implicit relationship)
        self.gpu_enabled = self.wal_enabled
        self.debug = debug
        self.logger = get_logger("GPUWarmTier")
        
        # Throttling for memory warnings to reduce spam
        self._last_memory_warning_time = 0

        self.gpu_available = GPU_AVAILABLE
        if not self.gpu_available:
            self.logger.warning("GPU not available - warm tier will operate in pass-through mode.")

        # Calculate max records for preallocated arrays with realistic sizing
        # Conservative estimate: partition should use max 25% of total GPU memory
        # Estimate: 8 columns * 8 bytes per column = 64 bytes per record
        partition_memory_limit = (self.max_memory_mb * 0.25) * 1024 * 1024  # 25% of total memory per partition
        bytes_per_record = 64
        self.max_records_per_partition = min(500_000, int(partition_memory_limit // bytes_per_record))
        
        self.logger.info(f"GPU Warm Tier: Will preallocate arrays for {self.max_records_per_partition:,} records per partition")
        
        # Preallocated GPU arrays and tracking
        self.gpu_arrays: Dict[str, Dict[str, cp.ndarray]] = {}  # partition -> column -> preallocated array
        self.partition_sizes: Dict[str, int] = {}  # partition -> current record count
        self.string_dictionaries: Dict[str, Dict[str, dict]] = {}  # partition -> column -> hash_to_string_map
        
        # Keep old interfaces for compatibility with existing managers
        self.gpu_data = self.gpu_arrays  # Alias for existing code
        self.gpu_metadata: Dict[str, dict] = {}
        self.partition_schemas: Dict[str, pa.Schema] = {}  # Store original schemas
        self.total_records = 0  # Track total records in warm tier for proper statistics
        
        # WAL Manager
        warm_wal_path = Path(wal_dir) / "warm"
        self.wal_manager = WALManager(warm_wal_path, "warm")
        
        # Component Managers
        self.eviction_callback: Optional[EvictionCallback] = None
        self.cache_manager = GPUCacheManager(
            self.gpu_data, self.gpu_metadata, self.string_dictionaries, 
            debug, config, self.partition_sizes, self.max_records_per_partition, self.max_memory_mb,
            self.partition_schemas
        )
        self.query_handler = GPUQueryHandler(self.gpu_data, self.gpu_metadata, self.string_dictionaries, debug, config)
        # Give query handler access to partition sizes for accurate filtering
        self.query_handler.partition_sizes = self.partition_sizes
        self.eviction_manager = GPUEvictionManager(
            self.gpu_data, self.gpu_metadata, self.string_dictionaries,
            self.wal_manager, self.eviction_callback, self.max_memory_mb, debug,
            eviction_threshold_pct=getattr(self, 'eviction_threshold_pct', 0.8), config=config,
            partition_sizes=self.partition_sizes, partition_schemas=self.partition_schemas
        )
        
        # Clear any existing GPU memory to start fresh
        self._clear_gpu_memory()
        
        self._recover_from_wal()

    def set_eviction_callback(self, callback: EvictionCallback):
        """Sets the eviction callback and updates the eviction manager."""
        self.eviction_callback = callback
        self.eviction_manager.eviction_callback = callback

    async def ingest(self, batch: pa.RecordBatch) -> bool:
        """Ingests data, writing to WAL and caching to GPU."""
        if not self.gpu_available:
            return await self._pass_through_ingest(batch)

        try:
            self.wal_manager.write_batch(batch)
        except Exception as e:
            self.logger.error(f"WAL write failed with schema error: {e}")
            self.logger.error(f"Batch schema: {batch.schema}")
            # Continue without WAL - the data can still be cached to GPU
            self.logger.warning("Continuing without WAL protection due to schema mismatch")
        
        success = await self.cache_manager.cache_to_gpu(batch)
        if success == "evict_needed":
            # Preallocation failed due to memory limits - trigger aggressive eviction
            current_memory_mb = self.cache_manager.calculate_gpu_memory_usage()  # Use actual data memory
            self.logger.info(f"Partition preallocation failed - triggering eviction. Actual data memory: {current_memory_mb:.1f}MB")
            evicted_count = await self.eviction_manager.evict_if_needed(current_memory_mb, aggressive=True)
            if evicted_count > 0:
                self.total_records -= evicted_count
                self.logger.info(f"WARM TIER COUNT DECREMENTED by {evicted_count:,} evicted records (count now: {self.total_records:,})")
            
            # Try to cache again after eviction
            success = await self.cache_manager.cache_to_gpu(batch)
            if success != True:
                self.logger.debug(f"GPU caching still failed after eviction - data in WAL only")
                success = False  # Convert to boolean for remaining logic
        
        if not success:
            # If caching failed due to memory pressure, try aggressive eviction and retry
            current_memory_mb = self.cache_manager.calculate_gpu_memory_usage()
            
            # Throttle memory warnings to avoid spam (max once per 10 seconds)
            import time
            current_time = time.time()
            if current_time - self._last_memory_warning_time > 10:
                # Fix: Calculate actual record count from GPU array lengths, not dict size
                total_records = 0
                for partition_data in self.gpu_data.values():
                    if partition_data:  # Check if partition has data
                        # Get length from first column (all columns should have same length)
                        first_column = next(iter(partition_data.values()))
                        total_records += len(first_column)
                self.logger.info(f"GPU caching failed due to memory pressure: {current_memory_mb:.1f}MB/{self.max_memory_mb}MB, Records: {total_records:,}")
                self._last_memory_warning_time = current_time
            else:
                self.logger.debug(f"GPU caching failed - memory limit reached")
                
            # Aggressive eviction to make space
            evicted_count = await self.eviction_manager.evict_if_needed(current_memory_mb, aggressive=True)
            if evicted_count > 0:
                self.total_records -= evicted_count
                self.logger.info(f"WARM TIER COUNT DECREMENTED by {evicted_count:,} evicted records (count now: {self.total_records:,})")
            
            # Retry caching after eviction - try multiple times
            retry_count = 0
            max_retries = 3
            while not success and retry_count < max_retries:
                success = await self.cache_manager.cache_to_gpu(batch)
                if not success:
                    retry_count += 1
                    if retry_count < max_retries:
                        # Force more aggressive eviction on retries
                        current_memory_mb = self.cache_manager.calculate_gpu_memory_usage()
                        evicted_count = await self.eviction_manager.evict_if_needed(current_memory_mb * 0.5, aggressive=True)  # Target 50% usage
                        if evicted_count > 0:
                            self.total_records -= evicted_count
                            self.logger.debug(f"Decremented warm tier count by {evicted_count:,} evicted records on retry")
                        self.logger.debug(f"Warm tier retry {retry_count}/{max_retries} after more aggressive eviction")
            
            # If still failed after retries, accept the batch anyway by writing to WAL
            # The data will be available for queries via WAL recovery even if not cached
            if not success:
                self.logger.debug(f"GPU caching failed after {max_retries} retries - data in WAL only")
        
        # If caching was successful, increment record count
        if success:
            self.total_records += batch.num_rows
            self.logger.info(f"WARM TIER COUNT INCREMENTED by {batch.num_rows:,} cached records (count now: {self.total_records:,})")
        
        # Always check for regular eviction after successful caching
        current_memory_mb = self.cache_manager.calculate_gpu_memory_usage()
        preallocated_memory_mb = self.cache_manager.calculate_preallocated_memory_usage()
        
        # Use actual data memory for eviction decisions - preallocated memory doesn't reflect actual usage
        evicted_count = await self.eviction_manager.evict_if_needed(current_memory_mb)
        if evicted_count > 0:
            self.total_records -= evicted_count
            self.logger.info(f"WARM TIER COUNT DECREMENTED by {evicted_count:,} evicted records during regular eviction (count now: {self.total_records:,})")
        
        # Return success only if data was successfully cached or stored in WAL
        # Critical: Don't return True unless data is safely stored somewhere!
        if success:
            # Data successfully cached to GPU and recorded in total_records
            return True
        else:
            # GPU caching failed - this means data is lost unless it's in WAL
            # The data should be queryable from WAL, but this is a degraded state
            self.logger.error(f"CRITICAL: GPU caching failed for {batch.num_rows:,} records - data loss possible!")
            
            # For now, still return True to avoid breaking the pipeline, but this needs fixing
            # TODO: Implement proper WAL-based fallback querying or reject the batch
            return False  # Return False to force hot tier to retry or handle failure

    async def query(self, filters: dict = None, limit: int = None, sort_by: str = None, ascending: bool = True) -> AsyncIterator[pa.RecordBatch]:
        """Queries data on the GPU."""
        if not self.gpu_available:
            async for _ in []: yield
            return

        async for batch in self.query_handler.query(filters=filters, limit=limit, sort_by=sort_by, ascending=ascending):
            yield batch

    async def get_stats(self) -> dict:
        """Returns statistics for the warm tier."""
        # Use properly tracked total_records for accurate statistics
        # This avoids double-counting during migrations
        wal_stats = self.wal_manager.get_stats()
        
        if not self.gpu_available:
            return {
                "tier_name": "gpu_warm", "total_records": 0, "gpu_memory_mb": 0.0,
                "gpu_memory_limit_mb": self.max_memory_mb, "memory_used_pct": 0.0,
                "gpu_available": False, "passthrough_mode": True, **wal_stats
            }

        current_memory_mb = self.cache_manager.calculate_gpu_memory_usage()
        return {
            "tier_name": "gpu_warm",
            "total_records": self.total_records,  # Use properly tracked total
            "total_partitions": len(self.gpu_data),
            "gpu_memory_mb": current_memory_mb,
            "gpu_memory_limit_mb": self.max_memory_mb,
            "memory_used_pct": (current_memory_mb / self.max_memory_mb) * 100 if self.max_memory_mb > 0 else 0,
            "gpu_available": True,
            "passthrough_mode": False,
            **wal_stats
        }

    async def cleanup(self):
        """Cleans up resources, such as closing the WAL manager."""
        self.wal_manager.close()
        self.logger.info("GPU Warm Tier cleanup complete.")

    def _clear_gpu_memory(self):
        """Clear all GPU memory to start fresh."""
        if not self.gpu_available:
            return
            
        try:
            # Clear data structures
            self.gpu_data.clear()
            self.gpu_metadata.clear()
            self.string_dictionaries.clear()
            self.partition_schemas.clear()
            
            # Force GPU memory cleanup
            import gc
            gc.collect()
            
            # Get CuPy memory pool and free unused blocks
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
            
            self.logger.info("GPU memory cleared successfully")
        except Exception as e:
            self.logger.warning(f"Error clearing GPU memory: {e}")

    def _recover_from_wal(self):
        """Recovers state from the WAL on startup."""
        if not self.gpu_available: return
        self.logger.info("Recovering warm tier state from WAL...")
        for batch in self.wal_manager.recover_all_batches():
            self.cache_manager._cache_to_gpu_sync(batch)
        
        # Ensure partition_sizes is in sync with gpu_metadata after recovery
        # and calculate total_records from recovered data
        synced_count = 0
        recovered_records = 0
        for partition_key, metadata in self.gpu_metadata.items():
            record_count = metadata.get('record_count', 0)
            old_count = self.partition_sizes.get(partition_key, 0)
            if old_count != record_count:
                self.partition_sizes[partition_key] = record_count
                self.logger.debug(f"Synced partition_sizes for {partition_key}: {old_count} -> {record_count} records")
                synced_count += 1
            recovered_records += record_count
        
        # Set total_records from WAL recovery
        self.total_records = recovered_records
        if recovered_records > 0:
            self.logger.info(f"Recovered {recovered_records:,} records from WAL, set as warm tier total")
        
        # Note: partition_schemas will be populated during the first batch ingestion for each partition
        # after recovery, since the original schema information is not stored in WAL/GPU metadata
        
        if synced_count > 0:
            self.logger.info(f"Synced {synced_count} partition sizes during WAL recovery")
        
        self.logger.info("WAL recovery complete.")

    async def _pass_through_ingest(self, batch: pa.RecordBatch) -> bool:
        """Directly evicts data to the next tier when GPU is not available."""
        if self.eviction_callback:
            return await self.eviction_callback.on_eviction([batch])
        self.logger.error("Pass-through ingest failed: no eviction callback set.")
        return False
