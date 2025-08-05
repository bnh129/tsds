"""
Handles query execution for the GPU warm tier.
"""
from typing import Dict, AsyncIterator
import pyarrow as pa
import numpy as np
from .logger import get_logger

try:
    import cupy as cp
except ImportError:
    cp = None

class GPUQueryHandler:
    """Executes queries against data stored on the GPU."""

    def __init__(self, gpu_data: Dict, gpu_metadata: Dict, string_dictionaries: Dict, debug: bool = False, config=None):
        self.gpu_data = gpu_data
        self.gpu_metadata = gpu_metadata
        self.string_dictionaries = string_dictionaries
        self.debug = debug
        self.config = config
        self.logger = get_logger("GPUQueryHandler")
        
        
        # Calculate optimal parallelism based on GPU capabilities
        if cp:
            try:
                device = cp.cuda.Device()
                with device:
                    multiprocessor_count = device.attributes['MultiProcessorCount']
                    # Use 2 workers per SM for good GPU utilization without oversaturation
                    self.max_gpu_workers = min(multiprocessor_count * 2, 32)  # Cap at 32 for memory reasons
                    # Create CUDA streams for concurrent GPU operations
                    self.gpu_streams = [cp.cuda.Stream() for _ in range(self.max_gpu_workers)]
                    self.logger.info(f"Initialized {self.max_gpu_workers} GPU workers with CUDA streams")
            except Exception as e:
                self.logger.warning(f"Could not initialize GPU streams: {e}")
                self.max_gpu_workers = 4  # Fallback
                self.gpu_streams = []
        else:
            self.max_gpu_workers = 1
            self.gpu_streams = []
    

    async def query(self, filters: dict = None, limit: int = None, sort_by: str = None, ascending: bool = True) -> AsyncIterator[pa.RecordBatch]:
        """
        Performs a query against the GPU data, applying filters, limits, and sorting.
        """
        # Use parallel GPU processing for top-K with limits
        if sort_by and limit:
            async for batch in self._parallel_gpu_query(filters, limit, sort_by, ascending):
                yield batch
            return
            
        # Fallback for unlimited sorted queries - use parallel approach but without early limit
        elif sort_by:
            async for batch in self._parallel_gpu_query(filters, None, sort_by, ascending):
                yield batch
            return
        
        # Parallel unsorted path - process all partitions concurrently
        async for batch in self._parallel_unsorted_query(filters, limit):
            yield batch

    async def _parallel_gpu_query(self, filters: dict, limit: int, sort_by: str, ascending: bool) -> AsyncIterator[pa.RecordBatch]:
        """
        Process multiple GPU partitions in parallel for maximum GPU utilization.
        Uses multiple workers and CUDA streams for true concurrent GPU processing.
        """
        import asyncio
        import concurrent.futures
        
        if not self.gpu_data:
            return
        
        # Create shared thread pool for all GPU operations to maximize parallelism
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_gpu_workers) as gpu_executor:
            partition_tasks = []
            stream_index = 0
            
            async def process_partition(partition_key: str, gpu_cols: dict, cuda_stream) -> pa.RecordBatch:
                """Process a single partition on GPU with dedicated CUDA stream."""
                try:
                    loop = asyncio.get_event_loop()
                    
                    def gpu_operations_with_stream():
                        """All GPU operations for this partition using dedicated CUDA stream."""
                        # Use dedicated CUDA stream for this partition's operations
                        with cuda_stream:
                            self.gpu_metadata[partition_key]['last_access'] += 1
                            
                            record_count = self.gpu_metadata[partition_key]['record_count']
                            
                            # Get actual partition size
                            if hasattr(self, 'partition_sizes') and partition_key in self.partition_sizes:
                                actual_record_count = self.partition_sizes[partition_key]
                            else:
                                actual_record_count = record_count
                            
                            if actual_record_count == 0:
                                return None
                            
                            # Build filter mask on GPU with stream
                            mask = self._build_filter_mask_with_stream(filters, partition_key, actual_record_count, cuda_stream)
                            
                            if mask is not None:
                                matching_indices = cp.where(mask)[0]
                                if len(matching_indices) == 0:
                                    return None
                            else:
                                matching_indices = cp.arange(actual_record_count)
                            
                            # Apply early limit to reduce GPU↔CPU transfer (only if limit specified)
                            if limit and len(matching_indices) > limit * 2:  # 2x buffer for sorting
                                matching_indices = matching_indices[:limit * 2]
                            
                            return self._reconstruct_batch_with_stream(matching_indices, partition_key, cuda_stream)
                    
                    # Execute GPU operations in shared thread pool with higher parallelism
                    result = await loop.run_in_executor(gpu_executor, gpu_operations_with_stream)
                    return result
                        
                except Exception as e:
                    self.logger.error(f"Error processing partition {partition_key}: {e}")
                    return None
            
            # Start all partition tasks in parallel with CUDA streams
            for partition_key, gpu_cols in self.gpu_data.items():
                if gpu_cols:  # Only process non-empty partitions
                    # Assign CUDA stream (round-robin if more partitions than streams)
                    cuda_stream = self.gpu_streams[stream_index % len(self.gpu_streams)] if self.gpu_streams else None
                    stream_index += 1
                    
                    task = asyncio.create_task(process_partition(partition_key, gpu_cols, cuda_stream))
                    partition_tasks.append(task)
            
            if not partition_tasks:
                return
            
            # Gather results from all partitions in parallel
            try:
                all_batches = await asyncio.gather(*partition_tasks, return_exceptions=True)
                
                # Filter out None results and exceptions
                valid_batches = []
                for batch in all_batches:
                    if isinstance(batch, pa.RecordBatch) and batch.num_rows > 0:
                        valid_batches.append(batch)
                    elif isinstance(batch, Exception):
                        self.logger.error(f"Partition processing failed: {batch}")
                
                if valid_batches:
                    # Combine and sort results using PyArrow's optimized operations
                    combined_table = pa.concat_tables([pa.Table.from_batches([batch]) for batch in valid_batches])
                    
                    # Apply sorting and limit
                    if sort_by in combined_table.schema.names:
                        order = 'ascending' if ascending else 'descending'
                        combined_table = combined_table.sort_by([(sort_by, order)])
                        if limit:
                            combined_table = combined_table.slice(0, limit)
                    
                    # Yield results in batches
                    for batch in combined_table.to_batches():
                        if batch.num_rows > 0:
                            yield batch
            
            except Exception as e:
                self.logger.error(f"Parallel GPU query failed: {e}")

    async def _parallel_unsorted_query(self, filters: dict, limit: int) -> AsyncIterator[pa.RecordBatch]:
        """
        Process partitions in parallel for unsorted queries with high-performance GPU parallelism.
        Yields results as soon as they're available for maximum throughput.
        """
        import asyncio
        import concurrent.futures
        
        if not self.gpu_data:
            return
        
        returned_records = 0
        
        # Use shared thread pool for maximum parallelism
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_gpu_workers) as gpu_executor:
            partition_tasks = []
            stream_index = 0
            
            async def process_partition_unsorted(partition_key: str, gpu_cols: dict, cuda_stream) -> pa.RecordBatch:
                """Process a single partition for unsorted query with dedicated CUDA stream."""
                try:
                    loop = asyncio.get_event_loop()
                    
                    def gpu_operations_with_stream():
                        """GPU operations with dedicated CUDA stream for parallelism."""
                        with cuda_stream if cuda_stream else cp.cuda.Stream():
                            self.gpu_metadata[partition_key]['last_access'] += 1
                            
                            record_count = self.gpu_metadata[partition_key]['record_count']
                            
                            # Get actual partition size
                            if hasattr(self, 'partition_sizes') and partition_key in self.partition_sizes:
                                actual_record_count = self.partition_sizes[partition_key]
                            else:
                                actual_record_count = record_count
                            
                            if actual_record_count == 0:
                                return None
                            
                            # Build filter mask on GPU with stream
                            mask = self._build_filter_mask_with_stream(filters, partition_key, actual_record_count, cuda_stream)
                            
                            if mask is not None:
                                matching_indices = cp.where(mask)[0]
                                if len(matching_indices) == 0:
                                    return None
                            else:
                                matching_indices = cp.arange(actual_record_count)
                            
                            # Apply limit per partition for unsorted queries
                            if limit and len(matching_indices) > limit:
                                matching_indices = matching_indices[:limit]
                            
                            return self._reconstruct_batch_with_stream(matching_indices, partition_key, cuda_stream)
                    
                    # Execute in shared thread pool for higher parallelism
                    return await loop.run_in_executor(gpu_executor, gpu_operations_with_stream)
                        
                except Exception as e:
                    self.logger.error(f"Error processing partition {partition_key} (unsorted): {e}")
                    return None
            
            # Start all partition tasks in parallel with CUDA streams
            for partition_key, gpu_cols in self.gpu_data.items():
                if gpu_cols:
                    # Assign CUDA stream (round-robin if more partitions than streams)
                    cuda_stream = self.gpu_streams[stream_index % len(self.gpu_streams)] if self.gpu_streams else None
                    stream_index += 1
                    
                    task = asyncio.create_task(process_partition_unsorted(partition_key, gpu_cols, cuda_stream))
                    partition_tasks.append(task)
            
            if not partition_tasks:
                return
            
            # Process results as they complete for immediate streaming
            try:
                # Collect all results and yield them
                all_batches = await asyncio.gather(*partition_tasks, return_exceptions=True)
                
                for batch in all_batches:
                    if limit and returned_records >= limit:
                        break
                        
                    if isinstance(batch, pa.RecordBatch) and batch.num_rows > 0:
                        # Apply global limit check
                        if limit and returned_records + batch.num_rows > limit:
                            remaining = limit - returned_records
                            # Slice the batch to fit the limit
                            batch_table = pa.Table.from_batches([batch])
                            batch_table = batch_table.slice(0, remaining)
                            batch = batch_table.to_batches()[0]
                        
                        yield batch
                        returned_records += batch.num_rows
                    elif isinstance(batch, Exception):
                        self.logger.error(f"Partition processing failed: {batch}")
            
            except Exception as e:
                self.logger.error(f"Parallel unsorted GPU query failed: {e}")

    def _build_filter_mask_with_stream(self, filters: dict, partition_key: str, record_count: int, cuda_stream) -> cp.ndarray:
        """Constructs a boolean mask on the GPU using CUDA streams for parallel execution."""
        if not filters:
            return None

        with cuda_stream if cuda_stream else cp.cuda.Stream():
            mask = cp.ones(record_count, dtype=bool)
            for col_name, value in filters.items():
                if col_name not in self.gpu_data[partition_key]:
                    continue
                
                # Slice GPU array to actual record count to avoid shape mismatch
                full_gpu_col = self.gpu_data[partition_key][col_name]
                gpu_col = full_gpu_col[:record_count]  # Only use actual data, not preallocated space
                is_dict_encoded = col_name in self.string_dictionaries.get(partition_key, {})

                if isinstance(value, dict): # Range filter
                    # Get configured time column name
                    time_column_name = self.config.schema.time_column if self.config else 'timestamp'
                    for op, filter_val in value.items():
                        if col_name == time_column_name:
                            # Convert datetime to int64 nanoseconds using PyArrow
                            ts_scalar = pa.scalar(filter_val, type=pa.timestamp('ns', tz='UTC'))
                            val_to_compare = pa.compute.cast(ts_scalar, pa.int64()).as_py()
                        else:
                            val_to_compare = filter_val
                        if op == '>=': mask &= (gpu_col >= val_to_compare)
                        elif op == '>': mask &= (gpu_col > val_to_compare)
                        elif op == '<=': mask &= (gpu_col <= val_to_compare)
                        elif op == '<': mask &= (gpu_col < val_to_compare)
                
                elif isinstance(value, list): # IN filter
                    if is_dict_encoded:
                        dictionary = self.string_dictionaries[partition_key][col_name]
                        # Use PyArrow compute.index_in for efficient lookups
                        value_array = pa.array(value)
                        indices_result = pa.compute.index_in(value_array, dictionary)
                        # Filter out null indices (values not found in dictionary)
                        valid_indices = pa.compute.drop_null(indices_result)
                        if len(valid_indices) > 0:
                            indices_np = valid_indices.to_numpy()
                            mask &= cp.isin(gpu_col, cp.asarray(indices_np))
                        else:
                            mask &= cp.zeros_like(gpu_col, dtype=bool)
                    else:
                        mask &= cp.isin(gpu_col, cp.asarray(value))

                else: # Equality filter
                    if is_dict_encoded:
                        dictionary = self.string_dictionaries[partition_key][col_name]
                        # Use PyArrow compute.index_in for efficient single value lookup
                        value_scalar = pa.scalar(value)
                        index_result = pa.compute.index_in(pa.array([value]), dictionary)
                        if not index_result[0].is_valid:
                            # Value not found in dictionary
                            mask &= cp.zeros_like(gpu_col, dtype=bool)
                        else:
                            index = index_result[0].as_py()
                            mask &= (gpu_col == index)
                    else:
                        mask &= (gpu_col == value)
            return mask

    def _reconstruct_batch_with_stream(self, indices: cp.ndarray, partition_key: str, cuda_stream) -> pa.RecordBatch:
        """Converts filtered GPU data back into a PyArrow RecordBatch using CUDA streams."""
        try:
            with cuda_stream if cuda_stream else cp.cuda.Stream():
                arrow_columns = []
                column_names = []
                gpu_cols = self.gpu_data[partition_key]

                # Convert indices to numpy once for all columns
                indices_np = cp.asnumpy(indices)

                for col_name, gpu_array in gpu_cols.items():
                    # Filter on GPU first, then transfer to CPU
                    filtered_gpu_data = gpu_array[indices]
                    cpu_data = cp.asnumpy(filtered_gpu_data)

                    # Get configured time column name
                    time_column_name = self.config.schema.time_column if self.config else 'timestamp'
                    if col_name == time_column_name:
                        # Create PyArrow timestamp array directly from int64 nanoseconds
                        arrow_array = pa.array(cpu_data, type=pa.int64())
                        arrow_array = pa.compute.cast(arrow_array, pa.timestamp('ns', tz='UTC'))
                    elif col_name in self.string_dictionaries.get(partition_key, {}):
                        dictionary = self.string_dictionaries[partition_key][col_name]
                        # Create dictionary array from indices and dictionary
                        arrow_array = pa.DictionaryArray.from_arrays(pa.array(cpu_data, type=pa.int32()), dictionary)
                    else:
                        # Let PyArrow infer the type from numpy array
                        arrow_array = pa.array(cpu_data)
                    
                    arrow_columns.append(arrow_array)
                    column_names.append(col_name)

                if not arrow_columns:
                    return None
                
                # Create schema from the actual arrow arrays for better type inference
                schema = pa.schema([pa.field(name, arr.type) for name, arr in zip(column_names, arrow_columns)])
                return pa.RecordBatch.from_arrays(arrow_columns, schema=schema)
        except Exception as e:
            self.logger.error(f"Failed to reconstruct batch from GPU for partition {partition_key}: {e}", exc_info=True)
            return None

    def _build_filter_mask(self, filters: dict, partition_key: str, record_count: int) -> cp.ndarray:
        """Constructs a boolean mask on the GPU to filter records."""
        if not filters:
            return None

        mask = cp.ones(record_count, dtype=bool)
        for col_name, value in filters.items():
            if col_name not in self.gpu_data[partition_key]:
                continue
            
            # Slice GPU array to actual record count to avoid shape mismatch
            full_gpu_col = self.gpu_data[partition_key][col_name]
            gpu_col = full_gpu_col[:record_count]  # Only use actual data, not preallocated space
            is_dict_encoded = col_name in self.string_dictionaries.get(partition_key, {})

            if isinstance(value, dict): # Range filter
                # Get configured time column name
                time_column_name = self.config.schema.time_column if self.config else 'timestamp'
                for op, filter_val in value.items():
                    if col_name == time_column_name:
                        # Convert datetime to int64 nanoseconds using PyArrow
                        ts_scalar = pa.scalar(filter_val, type=pa.timestamp('ns', tz='UTC'))
                        val_to_compare = pa.compute.cast(ts_scalar, pa.int64()).as_py()
                    else:
                        val_to_compare = filter_val
                    if op == '>=': mask &= (gpu_col >= val_to_compare)
                    elif op == '>': mask &= (gpu_col > val_to_compare)
                    elif op == '<=': mask &= (gpu_col <= val_to_compare)
                    elif op == '<': mask &= (gpu_col < val_to_compare)
            
            elif isinstance(value, list): # IN filter
                if is_dict_encoded:
                    dictionary = self.string_dictionaries[partition_key][col_name]
                    # Use PyArrow compute.index_in for efficient lookups
                    value_array = pa.array(value)
                    indices_result = pa.compute.index_in(value_array, dictionary)
                    # Filter out null indices (values not found in dictionary)
                    valid_indices = pa.compute.drop_null(indices_result)
                    if len(valid_indices) > 0:
                        indices_np = valid_indices.to_numpy()
                        mask &= cp.isin(gpu_col, cp.asarray(indices_np))
                    else:
                        mask &= cp.zeros_like(gpu_col, dtype=bool)
                else:
                    mask &= cp.isin(gpu_col, cp.asarray(value))

            else: # Equality filter
                if is_dict_encoded:
                    dictionary = self.string_dictionaries[partition_key][col_name]
                    # Use PyArrow compute.index_in for efficient single value lookup
                    value_scalar = pa.scalar(value)
                    index_result = pa.compute.index_in(pa.array([value]), dictionary)
                    if not index_result[0].is_valid:
                        # Value not found in dictionary
                        mask &= cp.zeros_like(gpu_col, dtype=bool)
                    else:
                        index = index_result[0].as_py()
                        mask &= (gpu_col == index)
                else:
                    mask &= (gpu_col == value)
        return mask

    def _reconstruct_batch(self, indices: cp.ndarray, partition_key: str) -> pa.RecordBatch:
        """Converts filtered GPU data back into a PyArrow RecordBatch."""
        try:
            arrow_columns = []
            column_names = []
            gpu_cols = self.gpu_data[partition_key]

            # Convert indices to numpy once for all columns
            indices_np = cp.asnumpy(indices)

            for col_name, gpu_array in gpu_cols.items():
                # Filter on GPU first, then transfer to CPU
                filtered_gpu_data = gpu_array[indices]
                cpu_data = cp.asnumpy(filtered_gpu_data)

                # Get configured time column name
                time_column_name = self.config.schema.time_column if self.config else 'timestamp'
                if col_name == time_column_name:
                    # Create PyArrow timestamp array directly from int64 nanoseconds
                    arrow_array = pa.array(cpu_data, type=pa.int64())
                    arrow_array = pa.compute.cast(arrow_array, pa.timestamp('ns', tz='UTC'))
                elif col_name in self.string_dictionaries.get(partition_key, {}):
                    dictionary = self.string_dictionaries[partition_key][col_name]
                    # Create dictionary array from indices and dictionary
                    arrow_array = pa.DictionaryArray.from_arrays(pa.array(cpu_data, type=pa.int32()), dictionary)
                else:
                    # Let PyArrow infer the type from numpy array
                    arrow_array = pa.array(cpu_data)
                
                arrow_columns.append(arrow_array)
                column_names.append(col_name)

            if not arrow_columns:
                return None
            
            # Create schema from the actual arrow arrays for better type inference
            schema = pa.schema([pa.field(name, arr.type) for name, arr in zip(column_names, arrow_columns)])
            return pa.RecordBatch.from_arrays(arrow_columns, schema=schema)
        except Exception as e:
            self.logger.error(f"Failed to reconstruct batch from GPU for partition {partition_key}: {e}", exc_info=True)
            return None
