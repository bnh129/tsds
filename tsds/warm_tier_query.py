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

    async def query(self, filters: dict = None, limit: int = None, sort_by: str = None, ascending: bool = True) -> AsyncIterator[pa.RecordBatch]:
        """
        Performs a query against the GPU data, applying filters, limits, and sorting.
        """
        # If sorting is requested, collect all data first then sort (simplified approach)
        if sort_by:
            all_batches = []
            
            for partition_key, gpu_cols in self.gpu_data.items():
                self.gpu_metadata[partition_key]['last_access'] += 1
                
                record_count = self.gpu_metadata[partition_key]['record_count']
                mask = self._build_filter_mask(filters, partition_key, record_count)

                if mask is not None:
                    matching_indices = cp.where(mask)[0]
                    if len(matching_indices) == 0:
                        continue
                else:
                    matching_indices = cp.arange(record_count)
                
                result_batch = self._reconstruct_batch(matching_indices, partition_key)
                if result_batch:
                    all_batches.append(result_batch)
            
            if all_batches:
                # Combine all batches and sort using PyArrow (more reliable)
                combined_table = pa.concat_tables([pa.Table.from_batches([batch]) for batch in all_batches])
                
                # Apply sorting if sort column exists
                if sort_by in combined_table.schema.names:
                    order = 'ascending' if ascending else 'descending'
                    combined_table = combined_table.sort_by([(sort_by, order)])
                
                # Apply limit after sorting
                if limit and combined_table.num_rows > limit:
                    combined_table = combined_table.slice(0, limit)
                
                # Yield sorted results in batches
                for batch in combined_table.to_batches():
                    if batch.num_rows > 0:
                        yield batch
            return
        
        # Original unsorted path for better performance when no sorting needed
        returned_records = 0
        for partition_key, gpu_cols in self.gpu_data.items():
            if limit and returned_records >= limit:
                break

            self.gpu_metadata[partition_key]['last_access'] += 1
            
            record_count = self.gpu_metadata[partition_key]['record_count']
            mask = self._build_filter_mask(filters, partition_key, record_count)

            if mask is not None:
                matching_indices = cp.where(mask)[0]
                if len(matching_indices) == 0:
                    continue
            else:
                matching_indices = cp.arange(record_count)

            if limit:
                remaining = limit - returned_records
                if len(matching_indices) > remaining:
                    matching_indices = matching_indices[:remaining]
            
            result_batch = self._reconstruct_batch(matching_indices, partition_key)
            if result_batch:
                yield result_batch
                returned_records += result_batch.num_rows

    def _build_filter_mask(self, filters: dict, partition_key: str, record_count: int) -> cp.ndarray:
        """Constructs a boolean mask on the GPU to filter records."""
        if not filters:
            return None

        mask = cp.ones(record_count, dtype=bool)
        for col_name, value in filters.items():
            if col_name not in self.gpu_data[partition_key]:
                continue
            
            gpu_col = self.gpu_data[partition_key][col_name]
            is_dict_encoded = col_name in self.string_dictionaries.get(partition_key, {})

            if isinstance(value, dict): # Range filter
                # Get configured time column name
                time_column_name = self.config.schema.time_column if self.config else 'timestamp'
                for op, filter_val in value.items():
                    val_to_compare = np.datetime64(filter_val).astype('datetime64[ns]').view('int64') if col_name == time_column_name else filter_val
                    if op == '>=': mask &= (gpu_col >= val_to_compare)
                    elif op == '>': mask &= (gpu_col > val_to_compare)
                    elif op == '<=': mask &= (gpu_col <= val_to_compare)
                    elif op == '<': mask &= (gpu_col < val_to_compare)
            
            elif isinstance(value, list): # IN filter
                if is_dict_encoded:
                    dictionary = self.string_dictionaries[partition_key][col_name].to_pylist()
                    indices = [dictionary.index(v) for v in value if v in dictionary]
                    mask &= cp.isin(gpu_col, cp.array(indices)) if indices else cp.zeros_like(gpu_col, dtype=bool)
                else:
                    mask &= cp.isin(gpu_col, cp.array(value))

            else: # Equality filter
                if is_dict_encoded:
                    dictionary = self.string_dictionaries[partition_key][col_name].to_pylist()
                    try:
                        index = dictionary.index(value)
                        mask &= (gpu_col == index)
                    except ValueError:
                        mask &= cp.zeros_like(gpu_col, dtype=bool)
                else:
                    mask &= (gpu_col == value)
        return mask

    def _reconstruct_batch(self, indices: cp.ndarray, partition_key: str) -> pa.RecordBatch:
        """Converts filtered GPU data back into a PyArrow RecordBatch."""
        try:
            arrow_columns = []
            column_names = []
            gpu_cols = self.gpu_data[partition_key]

            for col_name, gpu_array in gpu_cols.items():
                filtered_data = gpu_array[indices]
                cpu_data = cp.asnumpy(filtered_data)

                # Get configured time column name
                time_column_name = self.config.schema.time_column if self.config else 'timestamp'
                if col_name == time_column_name:
                    arrow_array = pa.array(cpu_data.astype('datetime64[ns]'), type=pa.timestamp('ns', tz='UTC'))
                elif col_name in self.string_dictionaries.get(partition_key, {}):
                    dictionary = self.string_dictionaries[partition_key][col_name]
                    arrow_array = pa.DictionaryArray.from_arrays(cpu_data, dictionary)
                else:
                    arrow_array = pa.array(cpu_data)
                
                arrow_columns.append(arrow_array)
                column_names.append(col_name)

            if not arrow_columns:
                return None
            
            schema = pa.schema([pa.field(name, arr.type) for name, arr in zip(column_names, arrow_columns)])
            return pa.RecordBatch.from_arrays(arrow_columns, schema=schema)
        except Exception as e:
            self.logger.error(f"Failed to reconstruct batch from GPU for partition {partition_key}: {e}", exc_info=True)
            return None
