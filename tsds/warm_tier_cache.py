"""
Manages caching data to the GPU for the warm tier.
Handles the conversion from PyArrow RecordBatches to CuPy arrays.
"""
from typing import Dict
import pyarrow as pa
from .logger import get_logger

try:
    import cupy as cp
except ImportError:
    cp = None

class GPUCacheManager:
    """Handles caching and data conversion for the GPU warm tier."""

    def __init__(self, gpu_data: Dict, gpu_metadata: Dict, string_dictionaries: Dict, debug: bool = False, config=None):
        self.gpu_data = gpu_data
        self.gpu_metadata = gpu_metadata
        self.string_dictionaries = string_dictionaries
        self.debug = debug
        self.config = config
        self.logger = get_logger("GPUCacheManager")

    async def cache_to_gpu(self, batch: pa.RecordBatch) -> bool:
        """Asynchronously cache a batch of data to GPU memory."""
        return self._cache_to_gpu_sync(batch)

    def _cache_to_gpu_sync(self, batch: pa.RecordBatch) -> bool:
        """
        Caches a batch to GPU memory, handling data type conversions.
        This is the core logic for moving data into the warm tier.
        """
        try:
            # Use configured time column name
            time_column_name = self.config.schema.time_column if self.config else 'timestamp'
            
            # Check if time column exists in batch
            if time_column_name not in batch.schema.names:
                self.logger.error(f"Configured time column '{time_column_name}' not found in batch schema: {batch.schema.names}")
                return False
                
            timestamp_col = batch.column(time_column_name)
            if len(timestamp_col) == 0:
                return False
            
            first_ts = timestamp_col[0].as_py()
            partition_key = first_ts.strftime("%Y-%m-%d")

            gpu_columns = {}
            for i, field in enumerate(batch.schema):
                column_name = field.name
                col_array = batch.column(i)

                if pa.types.is_string(col_array.type) or pa.types.is_dictionary(col_array.type):
                    dict_array = pa.compute.dictionary_encode(col_array)
                    indices = dict_array.indices.to_numpy()
                    dictionary = dict_array.dictionary
                    gpu_columns[column_name] = cp.asarray(indices, dtype=cp.int32)
                    
                    if partition_key not in self.string_dictionaries:
                        self.string_dictionaries[partition_key] = {}
                    self.string_dictionaries[partition_key][column_name] = dictionary

                elif pa.types.is_timestamp(col_array.type):
                    numpy_array = col_array.to_numpy().astype('datetime64[ns]').view('int64')
                    gpu_columns[column_name] = cp.asarray(numpy_array, dtype=cp.int64)
                else:
                    numpy_array = col_array.to_numpy()
                    gpu_columns[column_name] = cp.asarray(numpy_array)

            if partition_key in self.gpu_data:
                concatenation_failed = False
                for col_name, new_data in gpu_columns.items():
                    if col_name in self.gpu_data[partition_key]:
                        try:
                            # Get current GPU memory info
                            mempool = cp.get_default_memory_pool()
                            used_bytes = mempool.used_bytes()
                            total_bytes = mempool.total_bytes()
                            new_size = new_data.nbytes
                            
                            # If concatenation would likely cause OOM, trigger eviction first
                            # Only check if we actually have a reasonable total_bytes (not just initial allocation)
                            if total_bytes > 100 * 1024 * 1024 and used_bytes + new_size > total_bytes * 0.95:  # 95% threshold, only if >100MB total
                                # Return False to trigger eviction in the calling code
                                return False
                            
                            self.gpu_data[partition_key][col_name] = cp.concatenate([
                                self.gpu_data[partition_key][col_name], new_data
                            ])
                        except cp.cuda.memory.OutOfMemoryError as e:
                            # If concatenation fails, this partition is too large
                            self.logger.debug(f"GPU OOM during concatenation for partition {partition_key}, triggering eviction")
                            concatenation_failed = True
                            # Return False to trigger eviction in the calling code
                            return False
                    else:
                        self.gpu_data[partition_key][col_name] = new_data
                
                # Only update record count if all concatenations succeeded
                if not concatenation_failed:
                    self.gpu_metadata[partition_key]['record_count'] += batch.num_rows
            else:
                self.gpu_data[partition_key] = gpu_columns
                self.gpu_metadata[partition_key] = {
                    'record_count': batch.num_rows,
                    'last_access': 0,
                    'created_at': first_ts
                }
            
            return True
        except Exception as e:
            self.logger.error(f"GPU caching failed: {e}", exc_info=True)
            return False

    def calculate_gpu_memory_usage(self) -> float:
        """Calculate the total memory used by data on the GPU in MB."""
        total_bytes = sum(
            gpu_array.nbytes
            for partition_data in self.gpu_data.values()
            for gpu_array in partition_data.values()
        )
        return total_bytes / (1024 * 1024)
