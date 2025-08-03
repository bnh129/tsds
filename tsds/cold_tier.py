"""
Simple cold tier for persistent storage of evicted data.
Uses Arrow files for durability.
"""

import asyncio
import shutil
import time
from typing import List, AsyncIterator, Optional, Dict, Union
from pathlib import Path
from datetime import datetime, timedelta
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
from concurrent.futures import ThreadPoolExecutor
from .interfaces import StorageTier
from .logger import get_logger


class ColdTier(StorageTier):
    """Persistent storage for data evicted from warm tier."""
    
    def __init__(self, storage_path: str, index_columns: List[str] = None, debug: bool = False, 
                 index_batch_interval: float = 1.0, config=None):
        # Store config for later use
        self.config = config
        
        # Use config values if available, otherwise use parameters
        if config is not None:
            self.compression = config.cold_tier.compression
            self.index_batch_interval = config.cold_tier.index_batch_interval
            self.staging_enabled = config.cold_tier.staging_enabled
        else:
            self.compression = "snappy"  # Default fallback
            self.index_batch_interval = index_batch_interval
            self.staging_enabled = True
        
        # Time partitioning is always enabled for time-series data
        self.partition_by_time = True
            
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        if self.staging_enabled:
            self.staging_path = self.storage_path / "staging"
            self.staging_path.mkdir(exist_ok=True)
        else:
            self.staging_path = None
            
        self.file_counter = 0
        self.total_records = 0
        self.debug = debug
        self.logger = get_logger("ColdTier")
        
        # Indexing is completely disabled for maximum performance
        self.index_columns = []  # Always empty - no indexing
        
        if self.debug:
            print(f"ColdTier: Parquet storage initialized")
            print(f"ColdTier: Storage path: {self.storage_path}")
    
    async def ingest(self, data: Union[pa.RecordBatch, pa.Table]) -> bool:
        """Store data in cold storage using memory-efficient atomic transaction."""
        try:
            # Convert Table to batches and process individually to avoid memory explosion
            if isinstance(data, pa.Table):
                self.logger.info(f"Processing table with {data.num_rows:,} rows as individual batches")
                
                # Process table in larger batches for better performance
                batch_size = 200000  # Process 200K rows at a time (better I/O efficiency)
                for batch in data.to_batches(max_chunksize=batch_size):
                    success = await self._atomic_ingest_transaction(batch)
                    if not success:
                        return False
                
                return True
            else:
                # Single batch - use original flow
                return await self._atomic_ingest_transaction(data)
            
        except Exception as e:
            self.logger.error(f"Atomic ingestion failed: {e}")
            return False
    
    def _convert_to_arrow_filters(self, filters: dict) -> Optional[List]:
        """Convert filter dict to PyArrow predicate pushdown format."""
        if not filters:
            return None
            
        arrow_filters = []
        
        for col_name, value in filters.items():
            try:
                if isinstance(value, list):
                    # IN filter
                    arrow_filters.append((col_name, 'in', value))
                elif isinstance(value, dict):
                    # Range filters
                    for op, filter_value in value.items():
                        if op == '>=':
                            arrow_filters.append((col_name, '>=', filter_value))
                        elif op == '>':
                            arrow_filters.append((col_name, '>', filter_value))
                        elif op == '<=':
                            arrow_filters.append((col_name, '<=', filter_value))
                        elif op == '<':
                            arrow_filters.append((col_name, '<', filter_value))
                else:
                    # Equality filter
                    arrow_filters.append((col_name, '=', value))
            except Exception as e:
                # Log warning for skipped filters to aid debugging
                if self.debug:
                    print(f"ColdTier: Skipping unparseable filter {col_name}={value}: {e}")
                self.logger.warning(f"Failed to convert filter {col_name}={value} to Arrow format: {e}")
                continue
        
        return arrow_filters if arrow_filters else None
    
    async def _atomic_ingest_transaction(self, batch: pa.RecordBatch) -> bool:
        """
        Optimized atomic transaction for cold tier ingestion:
        1. Write data to staging directory (async)
        2. Atomic move to final location (async)
        """
        staging_file = None
        try:
            self.logger.debug(f"Starting atomic transaction for {batch.num_rows:,} records")
            # Extract timestamp for directory partitioning using configured time column
            time_column_name = self.config.schema.time_column if self.config else 'timestamp'
            timestamp_col = None
            
            for i, field in enumerate(batch.schema):
                if field.name == time_column_name:
                    timestamp_col = batch.column(i)
                    break
            
            # Use first timestamp for partitioning, or current time
            if timestamp_col and len(timestamp_col) > 0:
                first_ts = timestamp_col[0].as_py()
                if hasattr(first_ts, 'year'):
                    partition_date = first_ts
                else:
                    # Handle timestamp as epoch
                    partition_date = datetime.fromtimestamp(first_ts / 1e9)
            else:
                partition_date = datetime.now()
            
            # Create date-based directory structure
            year_month_day = f"year={partition_date.year}/month={partition_date.month:02d}/day={partition_date.day:02d}"
            partition_dir = self.storage_path / year_month_day
            partition_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename and paths
            filename = f"data_{self.file_counter:06d}.parquet"
            staging_file = self.staging_path / filename
            final_file = partition_dir / filename
            self.file_counter += 1
            
            # Step 1: Write to staging directory (async)
            self.logger.debug(f"Writing {batch.num_rows:,} records to staging file {staging_file}")
            table = pa.Table.from_batches([batch])
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: pq.write_table(table, staging_file, compression=self.compression)
            )
            self.logger.debug(f"Successfully wrote to staging")
            
            # Step 2: Atomic move from staging to final location (async)  
            self.logger.debug(f"Moving from staging to final location: {final_file}")
            await asyncio.get_event_loop().run_in_executor(
                None, shutil.move, str(staging_file), str(final_file)
            )
            
            self.total_records += batch.num_rows
            self.logger.info(f"Atomically stored {batch.num_rows:,} records to {year_month_day}/{filename}")
            
            return True
            
        except Exception as e:
            # Cleanup staging file on failure
            if staging_file and staging_file.exists():
                staging_file.unlink()
            self.logger.error(f"Atomic transaction failed: {e}")
            return False
    
    async def fast_lookup(self, column_name: str, value: str) -> List[pa.RecordBatch]:
        """Fast lookup - NOT SUPPORTED without indexing. Use query() instead."""
        if self.debug:
            print(f"ColdTier: fast_lookup() not supported - no indexing enabled")
            print(f"ColdTier: Use query() method instead for file scanning")
        return []
    
    async def flush_pending_index(self):
        """No indexing to flush - no-op for compatibility."""
        if self.debug:
            print("ColdTier: No indexing to flush")
    
    async def build_index_from_files(self) -> int:
        """No indexing - no-op for compatibility."""
        if self.debug:
            print("ColdTier: No indexing enabled - skipping background indexing")
        return 0
    
    async def _process_file_parallel(self, file_path: Path, filters: dict, arrow_filters: Optional[List]) -> Optional[pa.Table]:
        """Process a single file with filters in a thread pool executor."""
        try:
            # Run the file reading in a thread pool to avoid blocking the event loop
            def read_and_filter():
                # Use predicate pushdown for better performance
                table = pq.read_table(file_path, filters=arrow_filters)
                # Apply additional filters in memory to ensure correctness
                filtered_table = self._apply_filters(table, filters)
                return filtered_table if filtered_table.num_rows > 0 else None
            
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                result = await loop.run_in_executor(executor, read_and_filter)
                return result
                
        except Exception as e:
            if self.debug:
                print(f"ColdTier: Error reading {file_path}: {e}")
            self.logger.warning(f"Failed to process file {file_path}: {e}")
            return None

    def _apply_filters(self, table: pa.Table, filters: dict) -> pa.Table:
        """Apply a filter dictionary to a PyArrow Table."""
        if not filters:
            return table
        
        mask = None
        for col_name, value in filters.items():
            if col_name not in table.schema.names:
                continue

            col_array = table.column(col_name)
            
            if isinstance(value, dict):
                # Handle range filters (e.g., {"<=": val1, ">=": val2})
                for op, filter_val in value.items():
                    op_mask = None
                    
                    # Handle timestamp precision mismatch by casting filter_val to match column type
                    if pa.types.is_timestamp(col_array.type):
                        if hasattr(filter_val, 'timestamp'):  # datetime object
                            # Convert datetime to timestamp with matching precision
                            filter_val = pa.scalar(filter_val, type=col_array.type)
                        elif isinstance(filter_val, (str, pa.Scalar)):
                            # Parse string timestamp with matching precision
                            filter_val = pa.scalar(filter_val, type=col_array.type)
                    
                    if op == '>=':
                        op_mask = pc.greater_equal(col_array, filter_val)
                    elif op == '>':
                        op_mask = pc.greater(col_array, filter_val)
                    elif op == '<=':
                        op_mask = pc.less_equal(col_array, filter_val)
                    elif op == '<':
                        op_mask = pc.less(col_array, filter_val)
                    
                    if op_mask is not None:
                        mask = pc.and_(mask, op_mask) if mask is not None else op_mask

            elif isinstance(value, list):
                # Handle IN filter
                col_mask = pc.is_in(col_array, pa.array(value))
                mask = pc.and_(mask, col_mask) if mask is not None else col_mask
            else:
                # Handle equality filter
                col_mask = pc.equal(col_array, value)
                mask = pc.and_(mask, col_mask) if mask is not None else col_mask
        
        if mask is not None:
            return table.filter(mask)
        return table
    
    async def query(self, filters: dict = None, limit: int = None, sort_by: str = None, ascending: bool = True) -> AsyncIterator[pa.RecordBatch]:
        """Query cold tier files with partition pruning and predicate pushdown."""
        if self.debug:
            print(f"ColdTier: Querying with filters={filters}, limit={limit} (NO INDEX - file scanning)")
        
        # If sorting is requested, use optimized approach with chunked processing
        if sort_by:
            # Get all parquet files with potential partition pruning
            parquet_files = []
            
            # Look for partitioned data first (year=YYYY/month=MM/day=DD/*.parquet)
            for year_dir in sorted(self.storage_path.glob("year=*")):
                for month_dir in sorted(year_dir.glob("month=*")):
                    for day_dir in sorted(month_dir.glob("day=*")):
                        parquet_files.extend(sorted(day_dir.glob("*.parquet")))
            
            # Also scan for legacy files in root directory
            parquet_files.extend(sorted(self.storage_path.glob("*.parquet")))
            
            # Convert filters for PyArrow predicate pushDown
            arrow_filters = self._convert_to_arrow_filters(filters)
            
            # Get max concurrent files from config
            max_concurrent = self.config.query.max_concurrent_files if self.config else 10
            semaphore = asyncio.Semaphore(max_concurrent)
            
            # Strategy 1: If we have a small limit, use top-K approach with parallel processing
            if limit and limit <= 100000:  # Top-K optimization for small limits
                all_records = []
                
                # Helper function to process file for top-K
                async def process_file_for_topk(file_path):
                    async with semaphore:
                        try:
                            # Read only columns needed for sorting + filtering
                            columns_to_read = set()
                            if filters:
                                columns_to_read.update(filters.keys())
                            columns_to_read.add(sort_by)
                            
                            def read_minimal_columns():
                                # Read minimal columns first for faster I/O
                                table = pq.read_table(file_path, filters=arrow_filters, columns=list(columns_to_read))
                                filtered_table = self._apply_filters(table, filters)
                                
                                if filtered_table.num_rows > 0:
                                    # Convert to list of tuples (sort_value, file_path, row_index)  
                                    sort_column = filtered_table.column(sort_by).to_pylist()
                                    return [(sort_value, file_path, i) for i, sort_value in enumerate(sort_column)]
                                return []
                            
                            # Run in thread pool
                            loop = asyncio.get_event_loop()
                            with ThreadPoolExecutor() as executor:
                                return await loop.run_in_executor(executor, read_minimal_columns)
                            
                        except Exception as e:
                            if self.debug:
                                print(f"ColdTier: Error reading {file_path}: {e}")
                            return []
                
                # Process files in parallel for top-K
                tasks = [process_file_for_topk(file_path) for file_path in parquet_files]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Collect all records from parallel results
                for result in results:
                    if isinstance(result, list):
                        all_records.extend(result)
                
                if all_records:
                    # Sort all records and take top K
                    all_records.sort(key=lambda x: x[0], reverse=not ascending)
                    top_records = all_records[:limit]
                    
                    # Group by file to minimize I/O
                    file_indices = {}
                    for sort_value, file_path, row_idx in top_records:
                        if file_path not in file_indices:
                            file_indices[file_path] = []
                        file_indices[file_path].append(row_idx)
                    
                    # Read full data only for selected records
                    result_tables = []
                    for file_path, indices in file_indices.items():
                        try:
                            full_table = pq.read_table(file_path, filters=arrow_filters)
                            filtered_table = self._apply_filters(full_table, filters)
                            if filtered_table.num_rows > 0:
                                # Select only the rows we need
                                selected_table = filtered_table.take(indices)
                                result_tables.append(selected_table)
                        except Exception as e:
                            continue
                    
                    if result_tables:
                        combined_table = pa.concat_tables(result_tables)
                        # Final sort (should be fast since we have limited records)
                        if sort_by in combined_table.schema.names:
                            order = 'ascending' if ascending else 'descending'
                            combined_table = combined_table.sort_by([(sort_by, order)])
                        
                        # Yield results
                        for batch in combined_table.to_batches():
                            if batch.num_rows > 0:
                                yield batch
                return
                
            # Strategy 2: For large or unlimited queries, use chunked parallel processing
            else:
                all_tables = []
                max_memory_mb = 500  # Limit memory usage to 500MB
                current_memory_mb = 0
                
                # Process files in chunks to manage memory and parallelism
                chunk_size = max_concurrent
                for i in range(0, len(parquet_files), chunk_size):
                    chunk_files = parquet_files[i:i + chunk_size]
                    
                    # Process this chunk of files in parallel
                    tasks = [self._process_file_parallel(file_path, filters, arrow_filters) for file_path in chunk_files]
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # Process results from this chunk
                    for file_path, result in zip(chunk_files, results):
                        if isinstance(result, Exception):
                            if self.debug:
                                print(f"ColdTier: Error processing {file_path}: {result}")
                            continue
                            
                        if result is None or result.num_rows == 0:
                            continue
                        
                        # Estimate memory usage (rough approximation)
                        table_memory_mb = (result.nbytes) / (1024 * 1024)
                        
                        # If adding this table would exceed memory limit, process what we have
                        if current_memory_mb + table_memory_mb > max_memory_mb and all_tables:
                            # Sort and yield current batch
                            combined_table = pa.concat_tables(all_tables)
                            if sort_by in combined_table.schema.names:
                                order = 'ascending' if ascending else 'descending'
                                combined_table = combined_table.sort_by([(sort_by, order)])
                            
                            for batch in combined_table.to_batches():
                                if batch.num_rows > 0:
                                    yield batch
                            
                            # Reset for next chunk
                            all_tables = []
                            current_memory_mb = 0
                        
                        all_tables.append(result)
                        current_memory_mb += table_memory_mb
                
                # Process remaining tables
                if all_tables:
                    combined_table = pa.concat_tables(all_tables)
                    if sort_by in combined_table.schema.names:
                        order = 'ascending' if ascending else 'descending'
                        combined_table = combined_table.sort_by([(sort_by, order)])
                    
                    # Apply limit if specified
                    if limit and combined_table.num_rows > limit:
                        combined_table = combined_table.slice(0, limit)
                    
                    for batch in combined_table.to_batches():
                        if batch.num_rows > 0:
                            yield batch
            return
        
        # Original unsorted path with parallel processing
        returned_records = 0
        
        # Get all parquet files with potential partition pruning
        parquet_files = []
        
        # Look for partitioned data first (year=YYYY/month=MM/day=DD/*.parquet)
        for year_dir in sorted(self.storage_path.glob("year=*")):
            for month_dir in sorted(year_dir.glob("month=*")):
                for day_dir in sorted(month_dir.glob("day=*")):
                    parquet_files.extend(sorted(day_dir.glob("*.parquet")))
        
        # Also scan for legacy files in root directory
        parquet_files.extend(sorted(self.storage_path.glob("*.parquet")))
        
        # Convert filters for PyArrow predicate pushdown
        arrow_filters = self._convert_to_arrow_filters(filters)
        
        # Get max concurrent files from config
        max_concurrent = self.config.query.max_concurrent_files if self.config else 10
        semaphore = asyncio.Semaphore(max_concurrent)
        
        # Process files in parallel batches
        async def process_file_with_semaphore(file_path):
            async with semaphore:
                return await self._process_file_parallel(file_path, filters, arrow_filters)
        
        # Process files in chunks to avoid overwhelming memory
        chunk_size = max_concurrent * 2  # Process 2x concurrent limit at a time
        for i in range(0, len(parquet_files), chunk_size):
            if limit and returned_records >= limit:
                break
                
            chunk_files = parquet_files[i:i + chunk_size]
            
            # Process this chunk of files in parallel
            tasks = [process_file_with_semaphore(file_path) for file_path in chunk_files]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results and yield batches
            for file_path, result in zip(chunk_files, results):
                if limit and returned_records >= limit:
                    break
                    
                if isinstance(result, Exception):
                    if self.debug:
                        print(f"ColdTier: Error processing {file_path}: {result}")
                    continue
                    
                if result is None:
                    continue
                
                # Convert to batches and apply limit
                for batch in result.to_batches():
                    if limit and returned_records >= limit:
                        break
                    
                    if limit:
                        remaining = limit - returned_records
                        if batch.num_rows > remaining:
                            batch = batch.slice(0, remaining)
                    
                    if batch.num_rows > 0:
                        yield batch
                        returned_records += batch.num_rows
    
    async def get_stats(self) -> dict:
        """Get cold tier statistics."""
        parquet_files = []
        
        # Count partitioned files
        for year_dir in self.storage_path.glob("year=*"):
            for month_dir in year_dir.glob("month=*"):
                for day_dir in month_dir.glob("day=*"):
                    parquet_files.extend(day_dir.glob("*.parquet"))
        
        # Count legacy files
        parquet_files.extend(self.storage_path.glob("*.parquet"))
        
        file_count = len(parquet_files)
        total_size_mb = sum(f.stat().st_size for f in parquet_files) / (1024 * 1024)
        
        return {
            "tier_name": "cold",
            "total_records": self.total_records,
            "total_files": file_count,
            "storage_size_mb": total_size_mb,
            "indexed_records": 0,  # No indexing
            "index_size_mb": 0,  # No indexing
            "storage_path": str(self.storage_path),
            "index_db_path": None  # No indexing
        }
    
    async def close(self):
        """Close and cleanup resources."""
        if self.debug:
            print("ColdTier: Cleanup complete")
    
    def __del__(self):
        """Nothing to cleanup."""
        pass
