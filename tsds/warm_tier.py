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

        self.gpu_available = GPU_AVAILABLE
        if not self.gpu_available:
            self.logger.warning("GPU not available - warm tier will operate in pass-through mode.")

        # Data stores
        self.gpu_data: Dict[str, Dict[str, cp.ndarray]] = {}
        self.gpu_metadata: Dict[str, dict] = {}
        self.string_dictionaries: Dict[str, Dict[str, pa.Array]] = {}
        
        # WAL Manager
        warm_wal_path = Path(wal_dir) / "warm"
        self.wal_manager = WALManager(warm_wal_path, "warm")
        
        # Component Managers
        self.eviction_callback: Optional[EvictionCallback] = None
        self.cache_manager = GPUCacheManager(self.gpu_data, self.gpu_metadata, self.string_dictionaries, debug, config)
        self.query_handler = GPUQueryHandler(self.gpu_data, self.gpu_metadata, self.string_dictionaries, debug, config)
        self.eviction_manager = GPUEvictionManager(
            self.gpu_data, self.gpu_metadata, self.string_dictionaries,
            self.wal_manager, self.eviction_callback, self.max_memory_mb, debug,
            eviction_threshold_pct=getattr(self, 'eviction_threshold_pct', 0.8), config=config
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

        self.wal_manager.write_batch(batch)
        
        success = await self.cache_manager.cache_to_gpu(batch)
        if not success:
            # If caching failed due to memory pressure, try aggressive eviction and retry once
            self.logger.info("GPU caching failed, triggering aggressive eviction and retrying")
            current_memory_mb = self.cache_manager.calculate_gpu_memory_usage()
            await self.eviction_manager.evict_if_needed(current_memory_mb, aggressive=True)
            
            # Retry caching after eviction
            success = await self.cache_manager.cache_to_gpu(batch)
            
        if success:
            current_memory_mb = self.cache_manager.calculate_gpu_memory_usage()
            await self.eviction_manager.evict_if_needed(current_memory_mb)
        
        return success

    async def query(self, filters: dict = None, limit: int = None, sort_by: str = None, ascending: bool = True) -> AsyncIterator[pa.RecordBatch]:
        """Queries data on the GPU."""
        if not self.gpu_available:
            async for _ in []: yield
            return

        async for batch in self.query_handler.query(filters=filters, limit=limit, sort_by=sort_by, ascending=ascending):
            yield batch

    async def get_stats(self) -> dict:
        """Returns statistics for the warm tier."""
        total_records = sum(meta['record_count'] for meta in self.gpu_metadata.values())
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
            "total_records": total_records,
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
        self.logger.info("WAL recovery complete.")

    async def _pass_through_ingest(self, batch: pa.RecordBatch) -> bool:
        """Directly evicts data to the next tier when GPU is not available."""
        if self.eviction_callback:
            return await self.eviction_callback.on_eviction([batch])
        self.logger.error("Pass-through ingest failed: no eviction callback set.")
        return False
