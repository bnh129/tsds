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
                 max_memory_mb: int, debug: bool = False, eviction_threshold_pct: float = 0.8, config=None, partition_sizes: Dict = None, partition_schemas: Dict = None):
        self.gpu_data = gpu_data
        self.gpu_metadata = gpu_metadata
        self.string_dictionaries = string_dictionaries
        self.wal_manager = wal_manager
        self.eviction_callback = eviction_callback
        self.max_memory_mb = max_memory_mb
        self.eviction_threshold_pct = eviction_threshold_pct
        self.debug = debug
        self.config = config
        self.partition_sizes = partition_sizes or {}
        self.partition_schemas = partition_schemas or {}
        self.logger = get_logger("GPUEvictionManager")

    async def evict_if_needed(self, current_memory_mb: float, aggressive: bool = False):
        """Checks memory usage and triggers eviction if a threshold is passed. Returns evicted record count."""
        eviction_threshold = self.max_memory_mb * self.eviction_threshold_pct
        
        self.logger.info(f"EVICTION CHECK: current={current_memory_mb:.1f}MB, max={self.max_memory_mb}MB, threshold_pct={self.eviction_threshold_pct}, threshold={eviction_threshold:.1f}MB")
        
        # Use more aggressive threshold if requested (for OOM recovery)
        if aggressive:
            eviction_threshold = self.max_memory_mb * 0.6  # Target 60% instead of 80%
            self.logger.info(f"Aggressive eviction triggered due to memory pressure - new threshold: {eviction_threshold:.1f}MB")
            
        if current_memory_mb >= eviction_threshold:
            self.logger.info(f"✓ TRIGGERING EVICTION: {current_memory_mb:.1f}MB >= {eviction_threshold:.1f}MB")
            excess_mb = current_memory_mb - (self.max_memory_mb * 0.7)  # Target 70% usage
            return await self._evict_oldest_partitions(excess_mb)
        else:
            self.logger.info(f"✗ NO EVICTION: {current_memory_mb:.1f}MB < {eviction_threshold:.1f}MB")
            return 0

    async def _evict_oldest_partitions(self, excess_mb: float):
        """Evicts the least recently used partitions to free up space."""
        if not self.eviction_callback:
            self.logger.warning("Eviction needed but no eviction_callback is set.")
            return 0

        space_to_free_mb = max(excess_mb, self.max_memory_mb * 0.1) # Free at least 10%
        self.logger.info(f"Starting eviction: need to free {space_to_free_mb:.1f}MB")
        
        self.logger.debug(f"Eviction selection from {len(self.gpu_metadata)} GPU partitions, {len(self.partition_sizes)} tracked in partition_sizes")
        
        partitions_by_age = sorted(self.gpu_metadata.items(), key=lambda item: item[1]['last_access'])
        
        partitions_to_evict = []
        freed_mb = 0
        for partition_key, metadata in partitions_by_age:
            if freed_mb >= space_to_free_mb:
                break
            
            # Calculate actual memory usage using GPU array sizes (like the fixed memory calculation)
            partition_size_bytes = 0
            for gpu_array in self.gpu_data[partition_key].values():
                if hasattr(gpu_array, 'nbytes'):
                    partition_size_bytes += gpu_array.nbytes
            partition_size_mb = partition_size_bytes / (1024 * 1024)
            
            self.logger.debug(f"Partition {partition_key}: {partition_size_mb:.1f}MB")
            partitions_to_evict.append(partition_key)
            freed_mb += partition_size_mb

        self.logger.info(f"Selected {len(partitions_to_evict)} partitions for eviction, will free {freed_mb:.1f}MB")
        
        if not partitions_to_evict:
            self.logger.warning("No partitions selected for eviction")
            return 0

        eviction_batches = self._prepare_batches_for_eviction(partitions_to_evict)
        if not eviction_batches:
            return 0

        records_migrated = await self.eviction_callback.on_eviction(eviction_batches)
        if records_migrated > 0:
            self.logger.info(f"Successfully evicted {len(partitions_to_evict)} partitions.")
            # Clean up the evicted partitions from GPU memory
            self._cleanup_evicted_partitions(partitions_to_evict)
            self.logger.info(f"Evicted {records_migrated:,} records from {len(partitions_to_evict)} partitions")
            return records_migrated  # Return the actual migrated count from the callback
        else:
            self.logger.error("Eviction callback failed. Data remains on GPU.")
            return 0

    def _prepare_batches_for_eviction(self, partition_keys: List[str]) -> List[pa.RecordBatch]:
        """Converts GPU data for the given partitions back to Arrow RecordBatches."""
        batches = []
        for key in partition_keys:
            try:
                gpu_cols = self.gpu_data[key]
                if not gpu_cols:
                    continue
                    
                # Get actual record count from GPU metadata (not preallocated array size)
                if key in self.gpu_metadata:
                    actual_records = self.gpu_metadata[key].get('record_count', 0)
                else:
                    # Fallback: try partition_sizes
                    actual_records = self.partition_sizes.get(key, 0)
                
                if actual_records == 0:
                    continue
                    
                arrow_columns = []
                column_names = []
                for col_name, gpu_array in gpu_cols.items():
                    # Only convert actual data, not full preallocated array
                    actual_gpu_data = gpu_array[:actual_records]
                    cpu_data = cp.asnumpy(actual_gpu_data)
                    
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
                    # Use the original stored schema to maintain compatibility
                    original_schema = self.partition_schemas.get(key)
                    if original_schema:
                        # Ensure column order matches original schema
                        ordered_columns = []
                        ordered_names = []
                        for field in original_schema:
                            if field.name in column_names:
                                idx = column_names.index(field.name)
                                ordered_columns.append(arrow_columns[idx])
                                ordered_names.append(field.name)
                        
                        if ordered_columns:
                            batches.append(pa.RecordBatch.from_arrays(ordered_columns, schema=original_schema))
                    else:
                        # Fallback to dynamic schema if original not available
                        schema = pa.schema([pa.field(name, arr.type) for name, arr in zip(column_names, arrow_columns)])
                        batches.append(pa.RecordBatch.from_arrays(arrow_columns, schema=schema))
            except Exception as e:
                self.logger.error(f"Failed to prepare partition {key} for eviction: {e}", exc_info=True)
        return batches

    def _cleanup_evicted_partitions(self, partition_keys: List[str]):
        """Removes evicted partition data from GPU memory."""
        self.logger.debug(f"Cleaning up {len(partition_keys)} evicted partitions")
        
        for key in partition_keys:
            # Log the partition being cleaned up for debugging
            if key in self.partition_sizes:
                records_in_partition = self.partition_sizes[key]
                self.logger.debug(f"Cleaning up partition {key}: {records_in_partition:,} records")
            elif key in self.gpu_metadata:
                records_in_partition = self.gpu_metadata[key].get('record_count', 0)
                self.logger.debug(f"Cleaning up partition {key}: {records_in_partition:,} records (from gpu_metadata)")
            else:
                self.logger.warning(f"Cleaning up partition {key} - no record count available")
            
            # Remove all traces of this partition from GPU memory and tracking
            if key in self.gpu_data: del self.gpu_data[key]
            if key in self.gpu_metadata: del self.gpu_metadata[key]
            if key in self.string_dictionaries: del self.string_dictionaries[key]
            if key in self.partition_sizes: del self.partition_sizes[key]
            if key in self.partition_schemas: del self.partition_schemas[key]
        
        self.logger.debug(f"Cleaned up {len(partition_keys)} partitions from GPU memory")

