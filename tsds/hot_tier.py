"""
Simple in-memory hot tier with 1M record limit.
When full, triggers eviction to warm tier.
"""

import asyncio
from typing import List, AsyncIterator, Optional
from pathlib import Path
import pyarrow as pa
import pyarrow.compute as pc
from .interfaces import StorageTier, EvictionCallback
from .wal_manager import WALManager
from .logger import get_logger


class HotTier(StorageTier):
    """In-memory storage for recent data with 1M record limit."""
    
    def __init__(self, max_records: int = 1_000_000, wal_dir: str = "./wal", debug: bool = False, config=None):
        # Use config values if available, otherwise use parameters
        if config is not None:
            self.max_records = config.hot_tier.max_records
            self.eviction_threshold_pct = config.hot_tier.eviction_threshold_pct
            self.wal_enabled = config.hot_tier.wal_enabled
        else:
            self.max_records = max_records
            self.eviction_threshold_pct = 0.9  # Default fallback
            self.wal_enabled = True
        self.batches: List[pa.RecordBatch] = []
        self.total_records = 0
        self.eviction_callback: Optional[EvictionCallback] = None
        self.debug = debug
        self.logger = get_logger("HotTier")
        self.wal_transferred = False  # Track if WAL control was transferred
        
        # Initialize WAL manager
        self.wal_manager = WALManager(Path(wal_dir), "hot")
        
        # Recover from WAL on startup
        self._recover_from_wal()
    
    def set_eviction_callback(self, callback: EvictionCallback):
        """Set callback for when eviction is needed."""
        self.eviction_callback = callback
    
    def transfer_wal_control(self):
        """Mark that WAL control has been transferred to warm tier."""
        self.logger.info("WAL control transferred to warm tier")
        self.wal_transferred = True
    
    def _recover_from_wal(self):
        """Recover hot tier data from WAL segments."""
        if self.debug:
            print("HotTier: Recovering from WAL segments...")
        
        recovered_batches = list(self.wal_manager.recover_all_batches())
        
        if recovered_batches:
            self.batches = recovered_batches
            self.total_records = sum(batch.num_rows for batch in self.batches)
            
            if self.debug:
                print(f"HotTier: Recovered {len(recovered_batches)} batches, {self.total_records:,} records")
        else:
            if self.debug:
                print("HotTier: No WAL data to recover")
    
    async def ingest(self, batch: pa.RecordBatch) -> bool:
        """Ingest batch, evicting oldest data if necessary. Always accepts data."""
        # Check if we need to evict first - be more aggressive for LSM
        # Evict when we're at eviction threshold or would exceed limit
        eviction_threshold = int(self.max_records * self.eviction_threshold_pct)
        
        if (self.total_records >= eviction_threshold or 
            self.total_records + batch.num_rows > self.max_records):
            # Always try to evict, but continue even if eviction fails
            # The hot tier should never reject new data
            eviction_success = await self._evict_oldest_data()
            if not eviction_success:
                self.logger.error(f"Hot tier eviction failed - cannot safely accept new data to prevent data loss")
                # Return False to reject ingestion and preserve data integrity
                # This implements backpressure instead of data loss
                return False
        
        # Write to WAL first (durability) - only if we still control WAL
        if not self.wal_transferred:
            try:
                self.wal_manager.write_batch(batch)
                self.logger.debug(f"Wrote batch to WAL: {batch.num_rows:,} records")
            except Exception as e:
                self.logger.warning(f"WAL write failed: {e} - continuing without WAL protection")
                # Continue anyway - hot tier should always accept data even without WAL
        else:
            self.logger.debug(f"Skipping WAL write - control transferred to warm tier")
        
        # Add to in-memory storage after WAL write succeeds
        self.batches.append(batch)
        self.total_records += batch.num_rows
        return True
    
    async def _evict_oldest_data(self) -> bool:
        """Evict oldest 25% of data to make room."""
        if not self.eviction_callback or not self.batches:
            return False
        
        # Evict oldest 25% of batches
        evict_count = max(1, len(self.batches) // 4)
        batches_to_evict = self.batches[:evict_count]
        
        # Sort batches before eviction to create ordered data for LSM
        if len(batches_to_evict) > 1:
            try:
                # Combine batches and sort by timestamp for LSM efficiency
                combined_table = pa.Table.from_batches(batches_to_evict)
                if "timestamp" in combined_table.schema.names:
                    combined_table = combined_table.sort_by([("timestamp", "ascending")])
                    # Convert back to batches for eviction
                    batches_to_evict = list(combined_table.to_batches())
                    self.logger.debug(f"Sorted {len(batches_to_evict)} batches for LSM eviction")
            except Exception as e:
                self.logger.warning(f"Failed to sort batches for eviction: {e}")
                # Continue with unsorted batches
        
        # Mark WAL segments for potential disposal (transactional step 1)
        evictable_segments = self.wal_manager.get_evictable_segments(keep_latest=1)
        if evictable_segments:
            self.wal_manager.mark_segments_for_disposal(evictable_segments)
            if self.debug:
                print(f"HotTier: Marked {len(evictable_segments)} WAL segments for disposal: {evictable_segments}")
        
        # Try to evict to next tier
        migrated_records = await self.eviction_callback.on_eviction(batches_to_evict)
        
        if isinstance(migrated_records, bool):
            # Legacy boolean return - assume all records migrated if True
            if migrated_records:
                migrated_records = sum(b.num_rows for b in batches_to_evict)
            else:
                migrated_records = 0
        
        if migrated_records > 0:
            # Calculate how many batches were successfully migrated
            # We need to remove the right batches from memory
            total_to_evict = sum(b.num_rows for b in batches_to_evict)
            
            if migrated_records == total_to_evict:
                # All batches successfully migrated - remove all
                self.batches = self.batches[evict_count:]
                self.total_records -= migrated_records
                
                if self.debug:
                    print(f"HotTier: Successfully evicted all {migrated_records:,} records ({evict_count} batches)")
            else:
                # Partial migration - this is complex as we don't know which specific batches failed
                # For now, we'll keep all batches and only decrement the count
                # TODO: Implement proper partial eviction handling
                self.total_records -= migrated_records
                self.logger.warning(f"Partial eviction: {migrated_records:,}/{total_to_evict:,} records migrated. Keeping batches in memory.")
                
                if self.debug:
                    print(f"HotTier: Partial eviction {migrated_records:,}/{total_to_evict:,} records")
            
            # Notify the eviction callback which segments are safe to delete
            # This will be processed once WarmTier confirms durable storage
            if evictable_segments and migrated_records == total_to_evict:
                self.eviction_callback.confirm_wal_safe_to_delete(evictable_segments)
            
            # Clean up any confirmed safe segments
            cleaned_count = self.wal_manager.cleanup_confirmed_segments()
        else:
            if self.debug:
                print(f"HotTier: Eviction failed - no records migrated")
        
        return migrated_records > 0
    
    async def query(self, filters: dict = None, limit: int = None, sort_by: str = None, ascending: bool = True) -> AsyncIterator[pa.RecordBatch]:
        """Query hot tier data, with optional sorting."""
        all_results = []
        
        # Early exit for empty tier
        if not self.batches:
            return
        
        # Use Arrow's native optimized sort + slice for top-K
        if limit and sort_by:
            # Combine all batches
            combined_table = pa.Table.from_batches(self.batches)
            
            # Apply filters first
            if filters:
                mask = self._apply_filters(combined_table, filters)
                if mask is not None:
                    combined_table = combined_table.filter(mask)
            
            # Use Arrow's optimized sort + slice for top-K
            if sort_by in combined_table.schema.names:
                order = 'ascending' if ascending else 'descending'
                # Arrow's sort is highly optimized C++ code
                combined_table = combined_table.sort_by([(sort_by, order)])
                # Arrow's slice is O(1) operation
                combined_table = combined_table.slice(0, limit)
            
            # Yield results
            for batch in combined_table.to_batches():
                if batch.num_rows > 0:
                    yield batch
            return
        
        # Fallback: full processing for unlimited or unsorted queries
        combined_table = pa.Table.from_batches(self.batches)
        
        # Apply filters
        if filters:
            mask = self._apply_filters(combined_table, filters)
            if mask is not None:
                combined_table = combined_table.filter(mask)

        # Apply sorting
        if sort_by and sort_by in combined_table.schema.names:
            order = 'ascending' if ascending else 'descending'
            combined_table = combined_table.sort_by([(sort_by, order)])

        # Apply limit
        if limit:
            combined_table = combined_table.slice(0, limit)
            
        # Yield results in batches
        for batch in combined_table.to_batches():
            yield batch

    def _apply_filters(self, table: pa.Table, filters: dict) -> Optional[pa.Array]:
        """Apply a filter dictionary to a table and return the combined boolean mask."""
        if not filters:
            return None
        
        combined_mask = None

        for column, value in filters.items():
            if column not in table.schema.names:
                continue

            col_array = table.column(column)
            
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
                        combined_mask = pc.and_(combined_mask, op_mask) if combined_mask is not None else op_mask

            elif isinstance(value, list):
                # Handle IN filter
                col_mask = pc.is_in(col_array, pa.array(value))
                combined_mask = pc.and_(combined_mask, col_mask) if combined_mask is not None else col_mask
            else:
                # Handle equality filter
                col_mask = pc.equal(col_array, value)
                combined_mask = pc.and_(combined_mask, col_mask) if combined_mask is not None else col_mask
        
        return combined_mask
    
    async def get_stats(self) -> dict:
        """Get hot tier statistics."""
        # Only show WAL stats if we still control the WAL
        if hasattr(self, 'wal_transferred') and self.wal_transferred:
            # WAL control transferred to warm tier
            wal_segments = 0
            wal_size_mb = 0.0
        else:
            # We still have WAL control
            wal_stats = self.wal_manager.get_stats()
            wal_segments = wal_stats["segment_count"]
            wal_size_mb = wal_stats["total_size_mb"]
        
        # Calculate actual record count from batches instead of counter
        actual_records = sum(batch.num_rows for batch in self.batches)
        
        return {
            "tier_name": "hot",
            "total_records": actual_records,
            "total_batches": len(self.batches),
            "memory_usage_mb": sum(b.nbytes for b in self.batches) / (1024 * 1024),
            "capacity_used_pct": (self.total_records / self.max_records) * 100,
            "wal_segments": wal_segments,
            "wal_size_mb": wal_size_mb
        }
    
    async def cleanup(self):
        """Cleanup hot tier resources."""
        # Only close WAL if we still control it
        if not (hasattr(self, 'wal_transferred') and self.wal_transferred):
            self.wal_manager.close()
        if self.debug:
            print("HotTier: Cleanup complete")

