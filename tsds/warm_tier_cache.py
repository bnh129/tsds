"""
Manages caching data to the GPU for the warm tier.
PREALLOCATED VERSION: Uses fixed-size preallocated arrays for maximum performance.
"""
from typing import Dict
import pyarrow as pa
from .logger import get_logger

try:
    import cupy as cp
except ImportError:
    cp = None

class GPUCacheManager:
    """Handles caching to preallocated GPU arrays for the warm tier."""

    def __init__(self, gpu_data: Dict, gpu_metadata: Dict, string_dictionaries: Dict, debug: bool = False, config=None, partition_sizes: Dict = None, max_records_per_partition: int = None, max_memory_mb: int = 2048, partition_schemas: Dict = None):
        self.gpu_data = gpu_data
        self.gpu_metadata = gpu_metadata
        self.string_dictionaries = string_dictionaries
        self.debug = debug
        self.config = config
        self.logger = get_logger("GPUCacheManager")
        
        # References to warm tier's tracking variables
        self.partition_sizes = partition_sizes or {}
        self.partition_schemas = partition_schemas or {}
        self.max_records_per_partition = max_records_per_partition or 1000000
        self.max_memory_mb = max_memory_mb
        

    async def cache_to_gpu(self, batch: pa.RecordBatch) -> bool:
        """Asynchronously cache a batch to preallocated GPU arrays."""
        return self._cache_to_gpu_sync(batch)

    def _cache_to_gpu_sync(self, batch: pa.RecordBatch):
        """
        PREALLOCATED VERSION: Cache batch to preallocated GPU arrays.
        Much faster and more predictable than dynamic allocation.
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
            
            # Extract first timestamp scalar directly from array
            first_ts_scalar = timestamp_col[0]
            # Extract year, month, day using PyArrow compute functions
            year = pa.compute.year(first_ts_scalar)
            month = pa.compute.month(first_ts_scalar) 
            day = pa.compute.day(first_ts_scalar)
            partition_key = f"{year.as_py():04d}-{month.as_py():02d}-{day.as_py():02d}"
            
            # Initialize partition if first time
            if partition_key not in self.gpu_data:
                success = self._preallocate_partition(partition_key, batch.schema)
                if not success:
                    current_preallocated = self.calculate_preallocated_memory_usage()
                    self.logger.info(f"Failed to preallocate partition {partition_key} - current preallocated: {current_preallocated:.1f}MB, limit: {self.max_memory_mb * 0.9:.1f}MB")
                    # Signal that eviction is needed - the data will still be stored in WAL
                    return "evict_needed"  # Special return code to trigger eviction
                else:
                    # Store the original schema for this partition
                    self.partition_schemas[partition_key] = batch.schema
            else:
                # Partition exists, but ensure schema is stored (for WAL recovery cases)
                if partition_key not in self.partition_schemas:
                    self.partition_schemas[partition_key] = batch.schema
            
            # Get current size and check capacity  
            current_size = self.partition_sizes.get(partition_key, 0)
            max_records = self.max_records_per_partition
            
            if current_size + batch.num_rows > max_records:
                self.logger.debug(f"Partition {partition_key} full: {current_size}/{max_records} records")
                return False  # Trigger eviction
            
            # Copy data into preallocated arrays (much faster than concatenation)
            start_idx = current_size
            end_idx = start_idx + batch.num_rows
            
            for i, field in enumerate(batch.schema):
                column_name = field.name
                col_array = batch.column(i)
                
                if pa.types.is_string(col_array.type) or pa.types.is_dictionary(col_array.type):
                    # Use PyArrow's optimized dictionary encoding with safe incremental merging
                    dict_array = pa.compute.dictionary_encode(col_array)
                    indices = dict_array.indices.to_numpy()
                    dictionary = dict_array.dictionary
                    
                    # Store dictionary for this partition/column
                    if partition_key not in self.string_dictionaries:
                        self.string_dictionaries[partition_key] = {}
                    
                    if column_name not in self.string_dictionaries[partition_key]:
                        # First batch for this column - store the dictionary
                        self.string_dictionaries[partition_key][column_name] = dictionary
                    else:
                        # Subsequent batches - safe incremental dictionary building
                        existing_dict = self.string_dictionaries[partition_key][column_name]
                        
                        # Check if new strings are present (efficient set operation)
                        existing_strings = set(existing_dict.to_pylist())
                        new_strings_list = dictionary.to_pylist()
                        new_strings = [s for s in new_strings_list if s not in existing_strings]
                        
                        if new_strings:
                            # Only merge when new strings are found
                            self.logger.debug(f"Found {len(new_strings)} new strings in {column_name}, merging dictionary")
                            combined_dict = pa.concat_arrays([existing_dict, pa.array(new_strings)])
                            self.string_dictionaries[partition_key][column_name] = combined_dict
                            
                            # Re-encode current batch with the merged dictionary
                            dict_array = pa.DictionaryArray.from_arrays(
                                pa.compute.index_in(col_array, combined_dict),
                                combined_dict
                            )
                            indices = dict_array.indices.to_numpy()
                        # If no new strings, the original indices are still valid
                    
                    # Copy indices to GPU
                    self.gpu_data[partition_key][column_name][start_idx:end_idx] = cp.asarray(indices, dtype=cp.int32)
                
                elif pa.types.is_timestamp(col_array.type):
                    numpy_array = col_array.to_numpy().astype('datetime64[ns]').view('int64')
                    self.gpu_data[partition_key][column_name][start_idx:end_idx] = cp.asarray(numpy_array, dtype=cp.int64)
                
                else:
                    numpy_array = col_array.to_numpy()
                    self.gpu_data[partition_key][column_name][start_idx:end_idx] = cp.asarray(numpy_array)
            
            # Update size tracking
            new_size = current_size + batch.num_rows
            self.partition_sizes[partition_key] = new_size
            
            # Update metadata
            import time
            current_time = time.time()
            
            if partition_key not in self.gpu_metadata:
                self.gpu_metadata[partition_key] = {
                    'record_count': 0,
                    'last_access': current_time,  # Set to current time instead of 0
                    'created_at': first_ts_scalar.as_py()
                }
            
            self.gpu_metadata[partition_key]['record_count'] = new_size
            self.gpu_metadata[partition_key]['last_access'] = current_time  # Update on each write
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error caching to preallocated GPU arrays: {e}", exc_info=True)
            return False

    def _preallocate_partition(self, partition_key: str, schema: pa.Schema) -> bool:
        """Preallocate GPU arrays for a new partition. Returns True on success, False on failure."""
        if not cp:
            return False
        
        # Check if adding this partition would exceed memory limit
        current_preallocated_mb = self.calculate_preallocated_memory_usage()
        
        # Calculate how much memory this new partition would need
        max_records = self.max_records_per_partition
        bytes_per_partition = 0
        for field in schema:
            if pa.types.is_string(field.type) or pa.types.is_dictionary(field.type):
                bytes_per_partition += max_records * 4  # int32
            elif pa.types.is_timestamp(field.type):
                bytes_per_partition += max_records * 8  # int64
            elif pa.types.is_floating(field.type):
                bytes_per_partition += max_records * 8  # float64
            else:
                bytes_per_partition += max_records * 8  # int64 default
        
        new_partition_mb = bytes_per_partition / (1024 * 1024)
        total_after_allocation = current_preallocated_mb + new_partition_mb
        
        # Check against GPU memory limit - use 90% to leave some headroom
        memory_threshold = self.max_memory_mb * 0.9
        
        if total_after_allocation > memory_threshold:
            self.logger.debug(f"Cannot preallocate partition {partition_key}: would use {total_after_allocation:.1f}MB > {memory_threshold:.1f}MB limit")
            return False
            
        self.logger.debug(f"Preallocating GPU arrays for partition {partition_key}: {max_records:,} records ({new_partition_mb:.1f}MB)")
        
        try:
            self.gpu_data[partition_key] = {}
            
            for field in schema:
                column_name = field.name
                
                if pa.types.is_string(field.type) or pa.types.is_dictionary(field.type):
                    # Preallocate int32 array for string hashes
                    self.gpu_data[partition_key][column_name] = cp.zeros(max_records, dtype=cp.int32)
                elif pa.types.is_timestamp(field.type):
                    # Preallocate int64 array for timestamps
                    self.gpu_data[partition_key][column_name] = cp.zeros(max_records, dtype=cp.int64)
                elif pa.types.is_floating(field.type):
                    # Preallocate float64 array for floats
                    self.gpu_data[partition_key][column_name] = cp.zeros(max_records, dtype=cp.float64)
                else:
                    # Default to int64 for other types
                    self.gpu_data[partition_key][column_name] = cp.zeros(max_records, dtype=cp.int64)
            
            # Initialize partition size tracking
            self.partition_sizes[partition_key] = 0
            
            self.logger.debug(f"Preallocated {len(schema)} columns for partition {partition_key}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to preallocate partition {partition_key}: {e}")
            # Clean up partial allocation
            if partition_key in self.gpu_data:
                del self.gpu_data[partition_key]
            if partition_key in self.partition_sizes:
                del self.partition_sizes[partition_key]
            return False

    def calculate_gpu_memory_usage(self) -> float:
        """Calculate the total memory used by ACTUAL data in GPU arrays, not preallocated size."""
        if not cp:
            return 0.0
            
        total_bytes = 0
        for partition_key, partition_data in self.gpu_data.items():
            if isinstance(partition_data, dict):  # Skip partition_sizes etc
                # Use actual GPU array memory consumption
                for gpu_array in partition_data.values():
                    if hasattr(gpu_array, 'nbytes'):
                        total_bytes += gpu_array.nbytes
        
        return total_bytes / (1024 * 1024)

    def calculate_preallocated_memory_usage(self) -> float:
        """Calculate the total memory used by PREALLOCATED GPU arrays in MB."""
        if not cp:
            return 0.0
            
        total_bytes = 0
        for partition_data in self.gpu_data.values():
            if isinstance(partition_data, dict):  # Skip partition_sizes etc
                for gpu_array in partition_data.values():
                    if hasattr(gpu_array, 'nbytes'):
                        total_bytes += gpu_array.nbytes
        
        return total_bytes / (1024 * 1024)