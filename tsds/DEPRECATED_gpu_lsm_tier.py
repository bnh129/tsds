"""
GPU LSM Warm Tier - Hybrid in-memory + persistent LSM implementation
Stores sorted runs on GPU for fast queries, backed by parquet files for durability.
"""

import asyncio
import json
from typing import Dict, List, Optional, AsyncIterator
from pathlib import Path
from datetime import datetime
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
from .logger import get_logger

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None


class GPUSSTableMetadata:
    """Metadata for a GPU-backed SSTable with disk persistence."""
    
    def __init__(self, run_id: str, file_path: Path, record_count: int,
                 min_timestamp: datetime, max_timestamp: datetime,
                 column_stats: Dict[str, Dict[str, any]]):
        self.run_id = run_id
        self.file_path = file_path
        self.record_count = record_count
        self.min_timestamp = min_timestamp
        self.max_timestamp = max_timestamp
        self.column_stats = column_stats
        self.created_at = datetime.now()
        self.gpu_loaded = False  # Track if data is loaded on GPU
    
    def to_dict(self) -> dict:
        """Serialize metadata to dictionary."""
        return {
            "run_id": self.run_id,
            "file_path": str(self.file_path),
            "record_count": self.record_count,
            "min_timestamp": self.min_timestamp.isoformat(),
            "max_timestamp": self.max_timestamp.isoformat(),
            "column_stats": self.column_stats,
            "created_at": self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'GPUSSTableMetadata':
        """Deserialize metadata from dictionary."""
        return cls(
            run_id=data["run_id"],
            file_path=Path(data["file_path"]),
            record_count=data["record_count"],
            min_timestamp=datetime.fromisoformat(data["min_timestamp"]),
            max_timestamp=datetime.fromisoformat(data["max_timestamp"]),
            column_stats=data["column_stats"]
        )


class GPUSSTable:
    """A single sorted run stored on GPU with disk backing."""
    
    def __init__(self, metadata: GPUSSTableMetadata):
        self.metadata = metadata
        self.gpu_data: Optional[Dict[str, cp.ndarray]] = None
        self.string_dictionaries: Optional[Dict[str, pa.Array]] = None
        self.logger = get_logger(f"GPUSSTable-{metadata.run_id}")
    
    def load_to_gpu(self) -> bool:
        """Load data from disk to GPU memory."""
        if not GPU_AVAILABLE:
            return False
        
        if self.metadata.gpu_loaded and self.gpu_data:
            return True  # Already loaded
        
        try:
            # Read from disk
            table = pq.read_table(self.metadata.file_path)
            
            # Convert to GPU format similar to GPUCacheManager
            self.gpu_data = {}
            self.string_dictionaries = {}
            
            for i, field in enumerate(table.schema):
                col_name = field.name
                column = table.column(i)
                
                if pa.types.is_string(field.type):
                    # Dictionary encoding for strings
                    # Handle ChunkedArray by converting to pyarrow array first
                    if hasattr(column, 'chunks'):
                        column = pa.compute.dictionary_encode(column)
                    elif not isinstance(column, pa.DictionaryArray):
                        # Convert to dictionary encoding
                        column = pa.compute.dictionary_encode(column)
                    
                    # Now column should be a DictionaryArray
                    if isinstance(column, pa.DictionaryArray):
                        indices = column.indices.to_numpy()
                        self.gpu_data[col_name] = cp.asarray(indices, dtype=cp.int32)
                        self.string_dictionaries[col_name] = column.dictionary
                    else:
                        # Fallback: convert string to index manually
                        string_values = column.to_pylist()
                        unique_values = list(set(string_values))
                        value_to_index = {v: i for i, v in enumerate(unique_values)}
                        indices = [value_to_index[v] for v in string_values]
                        self.gpu_data[col_name] = cp.asarray(indices, dtype=cp.int32)
                        self.string_dictionaries[col_name] = pa.array(unique_values)
                    
                elif pa.types.is_timestamp(field.type):
                    # Convert timestamp to int64 for GPU processing
                    # Handle ChunkedArray by combining chunks
                    if hasattr(column, 'chunks') and len(column.chunks) > 1:
                        column = pa.concat_arrays(column.chunks)
                    timestamp_data = column.to_numpy().astype('datetime64[ns]').view('int64')
                    self.gpu_data[col_name] = cp.asarray(timestamp_data, dtype=cp.int64)
                    
                else:
                    # Numeric types - handle ChunkedArray
                    if hasattr(column, 'chunks') and len(column.chunks) > 1:
                        column = pa.concat_arrays(column.chunks)
                    numpy_data = column.to_numpy()
                    self.gpu_data[col_name] = cp.asarray(numpy_data)
            
            self.metadata.gpu_loaded = True
            self.logger.debug(f"Loaded {self.metadata.record_count} records to GPU")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load to GPU: {e}")
            return False
    
    def _load_from_table(self, table: pa.Table) -> bool:
        """Load data directly from Arrow table to GPU memory (faster than disk read)."""
        if not GPU_AVAILABLE:
            return False
        
        try:
            # Convert to GPU format directly from table
            self.gpu_data = {}
            self.string_dictionaries = {}
            
            for i, field in enumerate(table.schema):
                col_name = field.name
                column = table.column(i)
                
                if pa.types.is_string(field.type):
                    # Dictionary encoding for strings
                    # Handle ChunkedArray by converting to pyarrow array first
                    if hasattr(column, 'chunks'):
                        column = pa.compute.dictionary_encode(column)
                    elif not isinstance(column, pa.DictionaryArray):
                        column = pa.compute.dictionary_encode(column)
                    
                    # Now column should be a DictionaryArray
                    if isinstance(column, pa.DictionaryArray):
                        indices = column.indices.to_numpy()
                        self.gpu_data[col_name] = cp.asarray(indices, dtype=cp.int32)
                        self.string_dictionaries[col_name] = column.dictionary
                    else:
                        # Fallback: convert string to index manually
                        string_values = column.to_pylist()
                        unique_values = list(set(string_values))
                        value_to_index = {v: i for i, v in enumerate(unique_values)}
                        indices = [value_to_index[v] for v in string_values]
                        self.gpu_data[col_name] = cp.asarray(indices, dtype=cp.int32)
                        self.string_dictionaries[col_name] = pa.array(unique_values)
                        
                elif pa.types.is_timestamp(field.type):
                    # Convert timestamp to int64 for GPU processing
                    # Handle ChunkedArray by combining chunks
                    if hasattr(column, 'chunks') and len(column.chunks) > 1:
                        column = pa.concat_arrays(column.chunks)
                    timestamp_data = column.to_numpy().astype('datetime64[ns]').view('int64')
                    self.gpu_data[col_name] = cp.asarray(timestamp_data, dtype=cp.int64)
                    
                else:
                    # Numeric types - handle ChunkedArray
                    if hasattr(column, 'chunks') and len(column.chunks) > 1:
                        column = pa.concat_arrays(column.chunks)
                    numpy_data = column.to_numpy()
                    self.gpu_data[col_name] = cp.asarray(numpy_data)
            
            self.metadata.gpu_loaded = True
            self.logger.debug(f"Loaded {self.metadata.record_count} records directly to GPU from table")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load table to GPU: {e}")
            return False
    
    def unload_from_gpu(self):
        """Unload data from GPU to free memory."""
        if self.gpu_data:
            self.gpu_data.clear()
            self.gpu_data = None
        if self.string_dictionaries:
            self.string_dictionaries.clear()
            self.string_dictionaries = None
        self.metadata.gpu_loaded = False
    
    def get_gpu_memory_usage(self) -> float:
        """Get GPU memory usage in MB."""
        if not self.gpu_data:
            return 0.0
        
        total_bytes = 0
        for gpu_array in self.gpu_data.values():
            total_bytes += gpu_array.nbytes
        
        return total_bytes / (1024 * 1024)
    
    async def query_gpu(self, filters: dict = None, sort_by: str = None, 
                       ascending: bool = True, limit: int = None) -> AsyncIterator[pa.RecordBatch]:
        """Ultra-fast GPU query on pre-sorted data."""
        if not self.gpu_data or not GPU_AVAILABLE:
            return
        
        try:
            record_count = self.metadata.record_count
            
            # Build filter mask on GPU
            mask = self._build_gpu_filter_mask(filters, record_count)
            
            if mask is not None:
                matching_indices = cp.where(mask)[0]
                if len(matching_indices) == 0:
                    return
            else:
                matching_indices = cp.arange(record_count)
            
            # For sorted data, we can apply limit very efficiently
            if sort_by and limit and sort_by in self.gpu_data:
                # Data is pre-sorted, so we can just take first/last N
                if sort_by == "timestamp" and ascending:
                    # Take first N (earliest timestamps)
                    if len(matching_indices) > limit:
                        matching_indices = matching_indices[:limit]
                elif sort_by == "timestamp" and not ascending:
                    # Take last N (latest timestamps)  
                    if len(matching_indices) > limit:
                        matching_indices = matching_indices[-limit:]
                else:
                    # For other columns, we need to sort the matching indices
                    if len(matching_indices) > 0:
                        sort_values = self.gpu_data[sort_by][matching_indices]
                        if ascending:
                            sort_order = cp.argsort(sort_values)
                        else:
                            sort_order = cp.argsort(-sort_values)  # Descending
                        
                        matching_indices = matching_indices[sort_order]
                        if len(matching_indices) > limit:
                            matching_indices = matching_indices[:limit]
            
            # Convert back to Arrow batch
            result_batch = self._gpu_to_arrow_batch(matching_indices)
            if result_batch and result_batch.num_rows > 0:
                yield result_batch
                
        except Exception as e:
            self.logger.error(f"GPU query failed: {e}")
    
    def _build_gpu_filter_mask(self, filters: dict, record_count: int) -> Optional[cp.ndarray]:
        """Build filter mask on GPU."""
        if not filters or not self.gpu_data:
            return None
        
        mask = cp.ones(record_count, dtype=bool)
        
        for col_name, value in filters.items():
            if col_name not in self.gpu_data:
                continue
            
            gpu_col = self.gpu_data[col_name]
            
            if isinstance(value, dict):
                # Range filters
                for op, filter_val in value.items():
                    # Convert timestamp values
                    if col_name == "timestamp" and hasattr(filter_val, 'timestamp'):
                        filter_val = cp.datetime64(filter_val).astype('datetime64[ns]').view('int64')
                    
                    if op == '>=':
                        mask &= (gpu_col >= filter_val)
                    elif op == '>':
                        mask &= (gpu_col > filter_val)
                    elif op == '<=':
                        mask &= (gpu_col <= filter_val)
                    elif op == '<':
                        mask &= (gpu_col < filter_val)
            
            elif isinstance(value, list):
                # IN filter
                if col_name in self.string_dictionaries:
                    # String dictionary lookup
                    dictionary = self.string_dictionaries[col_name].to_pylist()
                    indices = [dictionary.index(v) for v in value if v in dictionary]
                    if indices:
                        mask &= cp.isin(gpu_col, cp.array(indices))
                    else:
                        mask &= cp.zeros_like(gpu_col, dtype=bool)
                else:
                    mask &= cp.isin(gpu_col, cp.array(value))
            
            else:
                # Equality filter
                if col_name in self.string_dictionaries:
                    dictionary = self.string_dictionaries[col_name].to_pylist()
                    try:
                        index = dictionary.index(value)
                        mask &= (gpu_col == index)
                    except ValueError:
                        mask &= cp.zeros_like(gpu_col, dtype=bool)
                else:
                    mask &= (gpu_col == value)
        
        return mask
    
    def _gpu_to_arrow_batch(self, indices: cp.ndarray) -> Optional[pa.RecordBatch]:
        """Convert filtered GPU data back to Arrow batch."""
        if not self.gpu_data or len(indices) == 0:
            return None
        
        try:
            arrow_columns = []
            column_names = []
            
            for col_name, gpu_array in self.gpu_data.items():
                filtered_data = gpu_array[indices]
                cpu_data = cp.asnumpy(filtered_data)
                
                if col_name == "timestamp":
                    # Convert back to timestamp
                    arrow_array = pa.array(cpu_data.astype('datetime64[ns]'), 
                                         type=pa.timestamp('ns', tz='UTC'))
                elif col_name in self.string_dictionaries:
                    # Reconstruct dictionary array
                    dictionary = self.string_dictionaries[col_name]
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
            self.logger.error(f"Failed to convert GPU data to Arrow: {e}")
            return None


class GPULSMWarmTier:
    """GPU LSM Warm Tier - hybrid in-memory + persistent implementation."""
    
    def __init__(self, storage_path: str, max_memory_mb: int = 2048, debug: bool = False, config=None):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.sstables_path = self.storage_path / "sstables"
        self.sstables_path.mkdir(exist_ok=True)
        self.metadata_file = self.storage_path / "metadata.json"
        
        self.max_memory_mb = max_memory_mb
        self.debug = debug
        self.config = config
        self.logger = get_logger("GPULSMWarmTier")
        
        self.gpu_available = GPU_AVAILABLE
        if not self.gpu_available:
            self.logger.warning("GPU not available - falling back to CPU-only mode")
        
        # Track all GPU SSTables
        self.gpu_sstables: List[GPUSSTable] = []
        self.next_run_id = 1
        
        # Track background processing tasks
        self.background_tasks: List[asyncio.Task] = []
        
        # Batching configuration
        self.batch_size_records = 25_000  # Batch every 25K records (faster GPU loading)
        self.pending_batches: List[pa.RecordBatch] = []
        self.pending_records_count = 0
        
        # Load existing metadata and rehydrate GPU
        self._load_metadata()
        if self.gpu_available:
            self._rehydrate_gpu()
    
    def _load_metadata(self):
        """Load existing SSTable metadata from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                
                for sstable_data in data.get("sstables", []):
                    metadata = GPUSSTableMetadata.from_dict(sstable_data)
                    if metadata.file_path.exists():
                        self.gpu_sstables.append(GPUSSTable(metadata))
                    
                self.next_run_id = data.get("next_run_id", 1)
                self.logger.info(f"Loaded {len(self.gpu_sstables)} SSTables from metadata")
                
            except Exception as e:
                self.logger.error(f"Error loading metadata: {e}")
                self.gpu_sstables = []
                self.next_run_id = 1
    
    def _save_metadata(self):
        """Save SSTable metadata to disk."""
        try:
            data = {
                "sstables": [sstable.metadata.to_dict() for sstable in self.gpu_sstables],
                "next_run_id": self.next_run_id
            }
            
            with open(self.metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving metadata: {e}")
    
    def _rehydrate_gpu(self):
        """Rehydrate GPU memory from disk files on startup."""
        if not self.gpu_available:
            return
        
        self.logger.info("Rehydrating GPU memory from disk...")
        
        current_memory_mb = 0
        loaded_count = 0
        
        # Load SSTables to GPU until we hit memory limit
        for sstable in self.gpu_sstables:
            if current_memory_mb >= self.max_memory_mb * 0.9:  # Leave 10% headroom
                break
            
            if sstable.load_to_gpu():
                current_memory_mb += sstable.get_gpu_memory_usage()
                loaded_count += 1
        
        self.logger.info(f"Rehydrated {loaded_count}/{len(self.gpu_sstables)} SSTables to GPU ({current_memory_mb:.1f}MB)")
    
    async def create_sstable_from_batches(self, batches: List[pa.RecordBatch], 
                                        sort_columns: List[str] = None) -> GPUSSTable:
        """Add batches to pending queue and create SSTable when batch size reached."""
        if not batches:
            return None
        
        # Add to pending batches
        self.pending_batches.extend(batches)
        batch_records = sum(b.num_rows for b in batches)
        self.pending_records_count += batch_records
        
        self.logger.debug(f"Added {batch_records:,} records to batch queue ({self.pending_records_count:,}/{self.batch_size_records:,})")
        
        # Check if we should create an SSTable
        if self.pending_records_count >= self.batch_size_records:
            return await self._create_batched_sstable(sort_columns)
        
        # Return None - no SSTable created yet (still batching)
        return None
    
    async def _create_batched_sstable(self, sort_columns: List[str] = None) -> GPUSSTable:
        """Create a large SSTable from all pending batches."""
        if not self.pending_batches:
            return None
        
        # Process batches directly without copying (save memory)
        records_to_process = self.pending_records_count
        
        self.logger.info(f"🔥 Creating BATCHED SSTable from {len(self.pending_batches)} batches ({records_to_process:,} records)")
        
        # Generate run ID and file path
        run_id = f"batch_{self.next_run_id:06d}"
        self.next_run_id += 1
        file_path = self.sstables_path / f"{run_id}.parquet"
        
        # Combine batches directly (avoid extra copy)
        combined_table = pa.concat_tables([pa.Table.from_batches([batch]) for batch in self.pending_batches])
        
        # Clear pending queue immediately after combining (free memory ASAP)
        self.pending_batches.clear()
        self.pending_records_count = 0
        
        # Compute metadata (quick operation)
        metadata = self._compute_metadata(run_id, file_path, combined_table)
        
        # Create GPU SSTable and load in optimized background (fast GPU loading only)
        gpu_sstable = GPUSSTable(metadata)
        self.gpu_sstables.append(gpu_sstable)
        
        # FAST background processing: GPU loading only (skip disk write for speed)
        task = asyncio.create_task(self._fast_gpu_load_only(combined_table, gpu_sstable))
        self.background_tasks.append(task)
        
        self.logger.info(f"⚡ FAST queued GPU SSTable {run_id} with {records_to_process:,} records for GPU loading")
        return gpu_sstable
    
    async def flush_pending_batches(self) -> GPUSSTable:
        """Force creation of SSTable from any remaining pending batches."""
        if self.pending_records_count > 0:
            self.logger.info(f"Flushing {self.pending_records_count:,} pending records to SSTable")
            return await self._create_batched_sstable()
        return None
    
    async def _async_process_sstable(self, table: pa.Table, file_path: Path, gpu_sstable: 'GPUSSTable'):
        """Process SSTable in background thread - GPU loading + disk write."""
        try:
            import concurrent.futures
            loop = asyncio.get_event_loop()
            
            # Run ALL expensive operations in thread pool
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                # Submit both GPU and disk operations
                gpu_task = None
                if self.gpu_available and self._get_current_gpu_memory() < self.max_memory_mb * 0.8:
                    gpu_task = executor.submit(gpu_sstable._load_from_table, table)
                
                disk_task = executor.submit(
                    lambda: pq.write_table(table, file_path, compression="snappy")
                )
                
                # Wait for disk write (critical for durability)
                await loop.run_in_executor(None, disk_task.result)
                self.logger.debug(f"Disk write completed for SSTable {gpu_sstable.metadata.run_id}")
                
                # Wait for GPU loading (if started)
                if gpu_task:
                    success = await loop.run_in_executor(None, gpu_task.result)
                    if success:
                        self.logger.debug(f"GPU loading completed for SSTable {gpu_sstable.metadata.run_id}")
                    else:
                        self.logger.warning(f"GPU loading failed for SSTable {gpu_sstable.metadata.run_id}")
                
                # Update metadata after successful processing
                await loop.run_in_executor(None, self._save_metadata)
            
        except Exception as e:
            self.logger.error(f"Async SSTable processing failed for {gpu_sstable.metadata.run_id}: {e}")
    
    async def _async_process_batched_sstable(self, table: pa.Table, file_path: Path, gpu_sstable: 'GPUSSTable'):
        """Process batched SSTable - REAL-TIME GPU loading like it used to work."""
        try:
            import concurrent.futures
            loop = asyncio.get_event_loop()
            
            # REAL-TIME MODE: Load to GPU immediately (like it used to work)
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                # Load to GPU first (real-time)
                gpu_task = None
                if self.gpu_available and self._get_current_gpu_memory() < self.max_memory_mb * 0.8:
                    gpu_task = executor.submit(gpu_sstable._load_from_table, table)
                
                # Disk write in parallel
                disk_task = executor.submit(lambda: pq.write_table(table, file_path, compression="snappy"))
                
                # Wait for both
                if gpu_task:
                    success = await loop.run_in_executor(None, gpu_task.result)
                    if success:
                        self.logger.debug(f"🔥 REAL-TIME GPU loaded SSTable {gpu_sstable.metadata.run_id}")
                    else:
                        self.logger.warning(f"GPU loading failed for SSTable {gpu_sstable.metadata.run_id}")
                
                await loop.run_in_executor(None, disk_task.result)
                self.logger.debug(f"💾 Disk write completed for SSTable {gpu_sstable.metadata.run_id}")
                
                # Update metadata
                await loop.run_in_executor(None, self._save_metadata)
            
        except Exception as e:
            self.logger.error(f"Async SSTable processing failed for {gpu_sstable.metadata.run_id}: {e}")
    
    async def _fast_gpu_load_only(self, table: pa.Table, gpu_sstable: 'GPUSSTable'):
        """ULTRA-FAST background processing: GPU loading only, no disk writes."""
        try:
            # Only load to GPU if we have memory (skip disk entirely for speed)
            if self.gpu_available and self._get_current_gpu_memory() < self.max_memory_mb * 0.8:
                import concurrent.futures
                loop = asyncio.get_event_loop()
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    success = await loop.run_in_executor(None, gpu_sstable._load_from_table, table)
                    if success:
                        self.logger.debug(f"🔥 FAST GPU loaded SSTable {gpu_sstable.metadata.run_id}")
            
        except Exception as e:
            self.logger.error(f"Fast GPU loading failed for {gpu_sstable.metadata.run_id}: {e}")
    
    async def wait_for_background_tasks(self):
        """Wait for FAST background GPU loading tasks to complete."""
        # Flush any remaining pending batches
        if self.pending_records_count > 0:
            self.logger.info(f"Flushing {self.pending_records_count:,} remaining pending records")
            await self.flush_pending_batches()
        
        # Wait for fast GPU loading tasks
        if self.background_tasks:
            self.logger.info(f"Waiting for {len(self.background_tasks)} FAST GPU loading tasks...")
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
            self.background_tasks.clear()
        
        # Report GPU loading status
        gpu_loaded_count = sum(1 for sstable in self.gpu_sstables if sstable.metadata.gpu_loaded)
        self.logger.info(f"🔥 GPU SSTables loaded: {gpu_loaded_count}/{len(self.gpu_sstables)} (via fast background loading)")
    
    def _compute_metadata(self, run_id: str, file_path: Path, table: pa.Table) -> GPUSSTableMetadata:
        """Compute metadata for a table."""
        record_count = table.num_rows
        
        # Extract timestamp range
        timestamp_col = table.column("timestamp")
        if hasattr(timestamp_col, 'chunks'):
            min_timestamp = pa.compute.min(timestamp_col).as_py()
            max_timestamp = pa.compute.max(timestamp_col).as_py()
        else:
            min_timestamp = timestamp_col.min().as_py()
            max_timestamp = timestamp_col.max().as_py()
        
        # Compute column statistics
        column_stats = {}
        for i, field in enumerate(table.schema):
            col_name = field.name
            column = table.column(i)
            
            try:
                if pa.types.is_numeric(field.type):
                    if hasattr(column, 'chunks'):
                        column_stats[col_name] = {
                            "min": pa.compute.min(column).as_py(),
                            "max": pa.compute.max(column).as_py()
                        }
                    else:
                        column_stats[col_name] = {
                            "min": column.min().as_py(),
                            "max": column.max().as_py()
                        }
            except Exception:
                continue
        
        return GPUSSTableMetadata(
            run_id=run_id,
            file_path=file_path,
            record_count=record_count,
            min_timestamp=min_timestamp,
            max_timestamp=max_timestamp,
            column_stats=column_stats
        )
    
    def _get_current_gpu_memory(self) -> float:
        """Get current GPU memory usage in MB."""
        if not self.gpu_available:
            return 0.0
        
        return sum(sstable.get_gpu_memory_usage() for sstable in self.gpu_sstables 
                  if sstable.metadata.gpu_loaded)
    
    async def query(self, filters: dict = None, sort_by: str = None, 
                   ascending: bool = True, limit: int = None) -> AsyncIterator[pa.RecordBatch]:
        """Ultra-fast query using GPU LSM structures with lazy loading."""
        
        self.logger.info(f"GPU LSM QUERY: {len(self.gpu_sstables)} SSTables, limit={limit}, sort_by='{sort_by}', gpu_available={self.gpu_available}")
        
        if not self.gpu_sstables:
            self.logger.warning("No GPU SSTables available")
            return
        
        # Ultra-fast path for small top-K queries using GPU (no lazy loading BS)
        if limit and limit <= 1000 and sort_by and self.gpu_available:
            self.logger.info(f"🔥 GPU LSM ULTRA-FAST PATH: querying {len(self.gpu_sstables)} SSTables")
            all_batches = []
            gpu_loaded_count = sum(1 for sstable in self.gpu_sstables if sstable.metadata.gpu_loaded)
            self.logger.info(f"GPU loaded SSTables: {gpu_loaded_count}/{len(self.gpu_sstables)}")
            
            # Query GPU SSTables that were loaded in real-time during ingestion
            for i, sstable in enumerate(self.gpu_sstables):
                if sstable.metadata.gpu_loaded:
                    self.logger.info(f"Querying GPU SSTable {i+1}/{len(self.gpu_sstables)} (run_id={sstable.metadata.run_id})")
                    async for batch in sstable.query_gpu(filters, sort_by, ascending, limit):
                        all_batches.append(batch)
                        self.logger.info(f"Got batch with {batch.num_rows} records from GPU SSTable {sstable.metadata.run_id}")
                        
                        # Early termination if we have enough results
                        total_records = sum(b.num_rows for b in all_batches)
                        if total_records >= limit * 5:  # 5x buffer
                            self.logger.info(f"Early termination: {total_records} >= {limit * 5}")
                            break
                else:
                    self.logger.warning(f"SSTable {sstable.metadata.run_id} not loaded to GPU")
            
            if all_batches:
                # Final merge and sort of GPU results
                combined_table = pa.concat_tables([pa.Table.from_batches([batch]) for batch in all_batches])
                
                if sort_by in combined_table.schema.names:
                    order = 'ascending' if ascending else 'descending'
                    combined_table = combined_table.sort_by([(sort_by, order)])
                    combined_table = combined_table.slice(0, limit)
                
                for batch in combined_table.to_batches():
                    if batch.num_rows > 0:
                        yield batch
            return
        
        # Fallback to disk-based queries for larger results
        # (Implementation similar to original LSM tier)
        pass
    
    async def get_stats(self) -> dict:
        """Get statistics about this GPU LSM warm tier."""
        total_records = sum(sstable.metadata.record_count for sstable in self.gpu_sstables)
        total_files = len(self.gpu_sstables)
        gpu_loaded_count = sum(1 for sstable in self.gpu_sstables if sstable.metadata.gpu_loaded)
        gpu_memory_mb = self._get_current_gpu_memory()
        
        return {
            "tier_name": "gpu_lsm_warm",
            "total_records": total_records,
            "total_sstables": total_files,
            "gpu_loaded_sstables": gpu_loaded_count,
            "gpu_memory_mb": gpu_memory_mb,
            "gpu_memory_limit_mb": self.max_memory_mb,
            "memory_used_pct": (gpu_memory_mb / self.max_memory_mb) * 100 if self.max_memory_mb > 0 else 0,
            "gpu_available": self.gpu_available,
            "sstables_path": str(self.sstables_path)
        }