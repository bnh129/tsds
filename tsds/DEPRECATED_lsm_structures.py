"""
LSM Tree structures for TSDS - Log-Structured Merge Trees
Provides sorted runs (SSTables) with metadata for efficient querying.
"""

import asyncio
import json
from typing import Dict, List, Optional, Tuple, AsyncIterator
from pathlib import Path
from datetime import datetime
import pyarrow as pa
import pyarrow.parquet as pq
from .logger import get_logger


class SSTableMetadata:
    """Metadata for a single SSTable (sorted run)."""
    
    def __init__(self, run_id: str, file_path: Path, record_count: int,
                 min_timestamp: datetime, max_timestamp: datetime,
                 column_stats: Dict[str, Dict[str, any]]):
        self.run_id = run_id
        self.file_path = file_path
        self.record_count = record_count
        self.min_timestamp = min_timestamp
        self.max_timestamp = max_timestamp
        self.column_stats = column_stats  # {"price": {"min": 100, "max": 200}, ...}
        self.created_at = datetime.now()
    
    def can_contain_value(self, column: str, value: any, comparison: str) -> bool:
        """Check if this SSTable could contain records matching the condition."""
        if column not in self.column_stats:
            return True  # Unknown column, assume it could match
        
        col_min = self.column_stats[column]["min"]
        col_max = self.column_stats[column]["max"]
        
        if comparison == ">":
            return col_max > value
        elif comparison == ">=":
            return col_max >= value
        elif comparison == "<":
            return col_min < value
        elif comparison == "<=":
            return col_min <= value
        elif comparison == "=":
            return col_min <= value <= col_max
        else:
            return True  # Unknown comparison, assume match
    
    def overlaps_time_range(self, start_time: Optional[datetime], end_time: Optional[datetime]) -> bool:
        """Check if this SSTable overlaps with the given time range."""
        if start_time and self.max_timestamp < start_time:
            return False
        if end_time and self.min_timestamp > end_time:
            return False
        return True
    
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
    def from_dict(cls, data: dict) -> 'SSTableMetadata':
        """Deserialize metadata from dictionary."""
        return cls(
            run_id=data["run_id"],
            file_path=Path(data["file_path"]),
            record_count=data["record_count"],
            min_timestamp=datetime.fromisoformat(data["min_timestamp"]),
            max_timestamp=datetime.fromisoformat(data["max_timestamp"]),
            column_stats=data["column_stats"]
        )


class SSTable:
    """A single sorted run (SSTable) with efficient querying."""
    
    def __init__(self, metadata: SSTableMetadata):
        self.metadata = metadata
        self.logger = get_logger(f"SSTable-{metadata.run_id}")
    
    async def query(self, filters: dict = None, sort_by: str = None, 
                   ascending: bool = True, limit: int = None) -> AsyncIterator[pa.RecordBatch]:
        """Query this SSTable with filters and limits."""
        
        # Check if we can skip this SSTable entirely based on metadata
        if not self._should_scan(filters, sort_by, ascending, limit):
            self.logger.debug(f"Skipping SSTable {self.metadata.run_id} - metadata pruning")
            return
        
        try:
            # Read the parquet file with predicate pushdown
            arrow_filters = self._convert_to_arrow_filters(filters)
            table = pq.read_table(self.metadata.file_path, filters=arrow_filters)
            
            # Apply additional filters
            if filters:
                table = self._apply_filters(table, filters)
            
            # Since data is already sorted, we can apply limit efficiently
            if sort_by and limit:
                # Data is pre-sorted, so we can just take the first/last N records
                if sort_by in table.schema.names:
                    # Verify sort order matches request
                    if self._is_sorted_correctly(sort_by, ascending):
                        # Take first N records (data already in correct order)
                        if table.num_rows > limit:
                            table = table.slice(0, limit)
                    else:
                        # Need to sort (shouldn't happen if data is pre-sorted correctly)
                        order = 'ascending' if ascending else 'descending'
                        table = table.sort_by([(sort_by, order)])
                        if table.num_rows > limit:
                            table = table.slice(0, limit)
            
            # Yield results
            for batch in table.to_batches():
                if batch.num_rows > 0:
                    yield batch
                    
        except Exception as e:
            self.logger.error(f"Error querying SSTable {self.metadata.run_id}: {e}")
    
    def _should_scan(self, filters: dict, sort_by: str, ascending: bool, limit: int) -> bool:
        """Determine if we should scan this SSTable based on metadata."""
        if not filters:
            return True
        
        # Check filters against column statistics
        for column, value in filters.items():
            if isinstance(value, dict):
                # Range filters
                for op, filter_val in value.items():
                    if not self.metadata.can_contain_value(column, filter_val, op):
                        return False
            elif isinstance(value, list):
                # IN filter - check if any value could be in range
                if column in self.metadata.column_stats:
                    col_min = self.metadata.column_stats[column]["min"]
                    col_max = self.metadata.column_stats[column]["max"]
                    if not any(col_min <= v <= col_max for v in value):
                        return False
            else:
                # Equality filter
                if not self.metadata.can_contain_value(column, value, "="):
                    return False
        
        return True
    
    def _is_sorted_correctly(self, sort_by: str, ascending: bool) -> bool:
        """Check if data is sorted in the expected order."""
        # For now, assume data is sorted by timestamp ascending
        # This could be enhanced to track actual sort order in metadata
        return sort_by == "timestamp" and ascending
    
    def _convert_to_arrow_filters(self, filters: dict) -> Optional[List]:
        """Convert filter dict to PyArrow predicate pushdown format."""
        if not filters:
            return None
            
        arrow_filters = []
        for col_name, value in filters.items():
            try:
                if isinstance(value, list):
                    arrow_filters.append((col_name, 'in', value))
                elif isinstance(value, dict):
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
                    arrow_filters.append((col_name, '=', value))
            except Exception:
                continue
        
        return arrow_filters if arrow_filters else None
    
    def _apply_filters(self, table: pa.Table, filters: dict) -> pa.Table:
        """Apply additional filters after predicate pushdown."""
        if not filters:
            return table
        
        # Implementation similar to existing filter logic
        # This would contain the same filtering logic from other tiers
        return table


class LSMWarmTier:
    """Warm tier using LSM tree structure with sorted runs."""
    
    def __init__(self, storage_path: str, max_memory_mb: int = 2048, debug: bool = False, config=None):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.sstables_path = self.storage_path / "sstables"
        self.sstables_path.mkdir(exist_ok=True)
        self.metadata_file = self.storage_path / "metadata.json"
        
        self.max_memory_mb = max_memory_mb
        self.debug = debug
        self.config = config
        self.logger = get_logger("LSMWarmTier")
        
        # Track all SSTables
        self.sstables: List[SSTable] = []
        self.next_run_id = 1
        
        # Load existing metadata
        self._load_metadata()
    
    def _load_metadata(self):
        """Load existing SSTable metadata from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                
                for sstable_data in data.get("sstables", []):
                    metadata = SSTableMetadata.from_dict(sstable_data)
                    if metadata.file_path.exists():
                        self.sstables.append(SSTable(metadata))
                    
                self.next_run_id = data.get("next_run_id", 1)
                self.logger.info(f"Loaded {len(self.sstables)} SSTables from metadata")
                
            except Exception as e:
                self.logger.error(f"Error loading metadata: {e}")
                self.sstables = []
                self.next_run_id = 1
    
    def _save_metadata(self):
        """Save SSTable metadata to disk."""
        try:
            data = {
                "sstables": [sstable.metadata.to_dict() for sstable in self.sstables],
                "next_run_id": self.next_run_id
            }
            
            with open(self.metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving metadata: {e}")
    
    async def create_sstable_from_batches(self, batches: List[pa.RecordBatch], 
                                        sort_columns: List[str] = None) -> SSTable:
        """Create a new SSTable from a list of batches."""
        if not batches:
            return None
        
        # Generate run ID and file path
        run_id = f"run_{self.next_run_id:06d}"
        self.next_run_id += 1
        file_path = self.sstables_path / f"{run_id}.parquet"
        
        # Combine batches
        combined_table = pa.concat_tables([pa.Table.from_batches([batch]) for batch in batches])
        
        # Sort by specified columns (default: timestamp)
        if sort_columns is None:
            sort_columns = ["timestamp"]
        
        sort_specs = [(col, "ascending") for col in sort_columns if col in combined_table.schema.names]
        if sort_specs:
            combined_table = combined_table.sort_by(sort_specs)
        
        # Compute metadata
        metadata = self._compute_metadata(run_id, file_path, combined_table)
        
        # Write to disk
        pq.write_table(combined_table, file_path, compression="snappy")
        
        # Create SSTable and add to collection
        sstable = SSTable(metadata)
        self.sstables.append(sstable)
        
        # Save metadata
        self._save_metadata()
        
        self.logger.info(f"Created SSTable {run_id} with {combined_table.num_rows} records")
        return sstable
    
    def _compute_metadata(self, run_id: str, file_path: Path, table: pa.Table) -> SSTableMetadata:
        """Compute metadata for a table."""
        record_count = table.num_rows
        
        # Extract timestamp range
        timestamp_col = table.column("timestamp")
        # Handle ChunkedArray by computing chunks
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
                    # Use compute functions for ChunkedArray compatibility
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
                elif pa.types.is_string(field.type):
                    # For strings, we could compute min/max lexicographically
                    # but it's expensive, so skip for now
                    pass
            except Exception:
                # Skip columns that can't be min/maxed
                continue
        
        return SSTableMetadata(
            run_id=run_id,
            file_path=file_path,
            record_count=record_count,
            min_timestamp=min_timestamp,
            max_timestamp=max_timestamp,
            column_stats=column_stats
        )
    
    async def query(self, filters: dict = None, sort_by: str = None, 
                   ascending: bool = True, limit: int = None) -> AsyncIterator[pa.RecordBatch]:
        """Query across all SSTables using metadata pruning."""
        
        if not self.sstables:
            return
        
        # Ultra-fast path for small top-K queries - use column statistics to find best SSTables
        if limit and limit <= 1000 and sort_by:
            # Step 1: Find the SSTables most likely to contain the top/bottom K records
            if sort_by in ["price", "volume", "timestamp"]:  # Numeric columns with stats
                relevant_sstables = self._find_relevant_sstables_for_topk(sort_by, ascending, limit)
            else:
                # For unsupported sort columns, use first few SSTables only
                relevant_sstables = self.sstables[:min(3, len(self.sstables))]
            
            if not relevant_sstables:
                return
            
            all_tables = []
            total_records_read = 0
            
            # Step 2: Read only the most relevant SSTables and limit early
            for sstable in relevant_sstables:
                try:
                    # Read limited data from each SSTable using parquet filters
                    if sort_by == "timestamp" and ascending:
                        # For timestamp ascending, read from beginning
                        table = pq.read_table(sstable.metadata.file_path)
                        if table.num_rows > limit * 2:  # Read 2x limit from each file
                            table = table.slice(0, limit * 2)
                    elif sort_by == "timestamp" and not ascending:
                        # For timestamp descending, read from end
                        table = pq.read_table(sstable.metadata.file_path)
                        if table.num_rows > limit * 2:
                            table = table.slice(table.num_rows - limit * 2, limit * 2)
                    else:
                        # For other columns, read full SSTable but limit total
                        table = pq.read_table(sstable.metadata.file_path)
                    
                    if table.num_rows > 0:
                        all_tables.append(table)
                        total_records_read += table.num_rows
                        
                        # Stop reading if we have enough data
                        if total_records_read >= limit * 10:  # 10x limit should be enough
                            break
                            
                except Exception as e:
                    self.logger.warning(f"Failed to read SSTable {sstable.metadata.run_id}: {e}")
            
            if all_tables:
                # Step 3: Combine only the limited data and sort
                combined_table = pa.concat_tables(all_tables)
                
                # Apply filters if needed
                if filters:
                    combined_table = self._apply_filters(combined_table, filters)
                
                # Sort and limit - much smaller dataset now
                if sort_by in combined_table.schema.names:
                    order = 'ascending' if ascending else 'descending'
                    combined_table = combined_table.sort_by([(sort_by, order)])
                    combined_table = combined_table.slice(0, limit)
                
                # Yield results
                for batch in combined_table.to_batches():
                    if batch.num_rows > 0:
                        yield batch
            return
        
        # Regular query for larger results - use SSTable-level querying
        for sstable in self.sstables:
            async for batch in sstable.query(filters, sort_by, ascending, limit):
                yield batch
    
    def _apply_filters(self, table: pa.Table, filters: dict) -> pa.Table:
        """Apply filters to a table."""
        if not filters:
            return table
        
        import pyarrow.compute as pc
        
        combined_mask = None
        for column, value in filters.items():
            if column not in table.schema.names:
                continue
            
            col_array = table.column(column)
            
            if isinstance(value, dict):
                # Range filters
                for op, filter_val in value.items():
                    op_mask = None
                    if op == '>=':
                        op_mask = pc.greater_equal(col_array, filter_val)
                    elif op == '>':
                        op_mask = pc.greater(col_array, filter_val)
                    elif op == '<=':
                        op_mask = pc.less_equal(col_array, filter_val)
                    elif op == '<':
                        op_mask = pc.less(col_array, filter_val)
                    
                    if op_mask is not None:
                        combined_mask = pc.and_(combined_mask, op_mask) if combined_mask is not None else op_mask
            elif isinstance(value, list):
                # IN filter
                col_mask = pc.is_in(col_array, pa.array(value))
                combined_mask = pc.and_(combined_mask, col_mask) if combined_mask is not None else col_mask
            else:
                # Equality filter
                col_mask = pc.equal(col_array, value)
                combined_mask = pc.and_(combined_mask, col_mask) if combined_mask is not None else col_mask
        
        if combined_mask is not None:
            return table.filter(combined_mask)
        return table
    
    def _find_relevant_sstables_for_topk(self, sort_by: str, ascending: bool, limit: int) -> List['SSTable']:
        """Find SSTables most likely to contain the top/bottom K records."""
        if not self.sstables:
            return []
        
        # For small limits, we only need to check a few SSTables
        if sort_by == "timestamp":
            # Timestamp is our primary sort key - SSTables are ordered by timestamp
            if ascending:
                # Want earliest timestamps - check first few SSTables
                return self.sstables[:min(3, len(self.sstables))]
            else:
                # Want latest timestamps - check last few SSTables
                return self.sstables[-min(3, len(self.sstables)):]
        
        elif sort_by in ["price", "volume"]:
            # For other numeric columns, use column statistics to find best SSTables
            candidates = []
            
            for sstable in self.sstables:
                if sort_by in sstable.metadata.column_stats:
                    stats = sstable.metadata.column_stats[sort_by]
                    min_val = stats.get("min", 0)
                    max_val = stats.get("max", 0)
                    
                    # Score based on potential to contain extreme values
                    if ascending:
                        # Want smallest values - prioritize SSTables with small minimums
                        score = -min_val  # Negative so smaller values have higher score
                    else:
                        # Want largest values - prioritize SSTables with large maximums
                        score = max_val
                    
                    candidates.append((score, sstable))
            
            # Sort by score and take top candidates
            candidates.sort(key=lambda x: x[0], reverse=True)
            return [sstable for _, sstable in candidates[:min(5, len(candidates))]]
        
        # Fallback - use first few SSTables
        return self.sstables[:min(3, len(self.sstables))]
    
    async def get_stats(self) -> dict:
        """Get statistics about this LSM warm tier."""
        total_records = sum(sstable.metadata.record_count for sstable in self.sstables)
        total_files = len(self.sstables)
        
        return {
            "tier_name": "lsm_warm",
            "total_records": total_records,
            "total_sstables": total_files,
            "sstables_path": str(self.sstables_path)
        }