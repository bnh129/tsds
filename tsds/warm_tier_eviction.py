"""
Manages data eviction from the GPU warm tier.
"""
from typing import Dict, List, Optional
import pyarrow as pa
from .interfaces import EvictionCallback
from .wal_manager import WALManager
from .logger import get_logger

try:
    import cupy as cp
except ImportError:
    cp = None

class GPUEvictionManager:
    """Handles eviction of data from GPU to the next tier."""

    def __init__(self, gpu_data: Dict, gpu_metadata: Dict, string_dictionaries: Dict, 
                 wal_manager: WALManager, eviction_callback: Optional[EvictionCallback], 
                 max_memory_mb: int, debug: bool = False, eviction_threshold_pct: float = 0.8, config=None):
        self.gpu_data = gpu_data
        self.gpu_metadata = gpu_metadata
        self.string_dictionaries = string_dictionaries
        self.wal_manager = wal_manager
        self.eviction_callback = eviction_callback
        self.max_memory_mb = max_memory_mb
        self.eviction_threshold_pct = eviction_threshold_pct
        self.debug = debug
        self.config = config
        self.logger = get_logger("GPUEvictionManager")

    async def evict_if_needed(self, current_memory_mb: float, aggressive: bool = False):
        """Checks memory usage and triggers eviction if a threshold is passed."""
        eviction_threshold = self.max_memory_mb * self.eviction_threshold_pct
        
        # Use more aggressive threshold if requested (for OOM recovery)
        if aggressive:
            eviction_threshold = self.max_memory_mb * 0.6  # Target 60% instead of 80%
            self.logger.info(f"Aggressive eviction triggered due to memory pressure")
            
        if current_memory_mb >= eviction_threshold:
            self.logger.info(f"Memory usage {current_memory_mb:.1f}MB exceeds threshold {eviction_threshold:.1f}MB. Triggering eviction.")
            excess_mb = current_memory_mb - (self.max_memory_mb * 0.7)  # Target 70% usage
            await self._evict_oldest_partitions(excess_mb)

    async def _evict_oldest_partitions(self, excess_mb: float):
        """Evicts the least recently used partitions to free up space."""
        if not self.eviction_callback:
            self.logger.warning("Eviction needed but no eviction_callback is set.")
            return

        space_to_free_mb = max(excess_mb, self.max_memory_mb * 0.1) # Free at least 10%
        
        partitions_by_age = sorted(self.gpu_metadata.items(), key=lambda item: item[1]['last_access'])
        
        partitions_to_evict = []
        freed_mb = 0
        for partition_key, metadata in partitions_by_age:
            if freed_mb >= space_to_free_mb:
                break
            
            partition_size_mb = sum(arr.nbytes for arr in self.gpu_data[partition_key].values()) / (1024*1024)
            partitions_to_evict.append(partition_key)
            freed_mb += partition_size_mb

        if not partitions_to_evict:
            return

        eviction_batches = self._prepare_batches_for_eviction(partitions_to_evict)
        if not eviction_batches:
            return

        success = await self.eviction_callback.on_eviction(eviction_batches)
        if success:
            self.logger.info(f"Successfully evicted {len(partitions_to_evict)} partitions.")
            self._cleanup_evicted_partitions(partitions_to_evict)
        else:
            self.logger.error("Eviction callback failed. Data remains on GPU.")

    def _prepare_batches_for_eviction(self, partition_keys: List[str]) -> List[pa.RecordBatch]:
        """Converts GPU data for the given partitions back to Arrow RecordBatches."""
        batches = []
        for key in partition_keys:
            try:
                gpu_cols = self.gpu_data[key]
                arrow_columns = []
                column_names = []
                for col_name, gpu_array in gpu_cols.items():
                    cpu_data = cp.asnumpy(gpu_array)
                    # Get configured time column name
                    time_column_name = self.config.schema.time_column if self.config else 'timestamp'
                    if col_name == time_column_name:
                        arrow_array = pa.array(cpu_data.astype('datetime64[ns]'), type=pa.timestamp('ns', tz='UTC'))
                    elif col_name in self.string_dictionaries.get(key, {}):
                        dictionary = self.string_dictionaries[key][col_name]
                        arrow_array = pa.DictionaryArray.from_arrays(cpu_data, dictionary)
                    else:
                        arrow_array = pa.array(cpu_data)
                    arrow_columns.append(arrow_array)
                    column_names.append(col_name)
                
                if arrow_columns:
                    schema = pa.schema([pa.field(name, arr.type) for name, arr in zip(column_names, arrow_columns)])
                    batches.append(pa.RecordBatch.from_arrays(arrow_columns, schema=schema))
            except Exception as e:
                self.logger.error(f"Failed to prepare partition {key} for eviction: {e}", exc_info=True)
        return batches

    def _cleanup_evicted_partitions(self, partition_keys: List[str]):
        """Removes evicted partition data from GPU memory."""
        for key in partition_keys:
            if key in self.gpu_data: del self.gpu_data[key]
            if key in self.gpu_metadata: del self.gpu_metadata[key]
            if key in self.string_dictionaries: del self.string_dictionaries[key]

