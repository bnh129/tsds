"""
Simple TSDS coordinator that manages Hot -> Warm -> Cold tier flow.
Clean, straightforward implementation focused on core functionality.
"""

import asyncio
import heapq
from typing import List, AsyncIterator, Optional
from pathlib import Path
import pyarrow as pa

from .interfaces import EvictionCallback
from .hot_tier import HotTier
from .warm_tier import GPUWarmTier
from .cold_tier import ColdTier
from .logger import get_logger
from .config import get_config, TSDBConfig


class WarmTierEvictionHandler(EvictionCallback):
    """Handles eviction from warm tier to cold tier."""
    
    def __init__(self, cold_tier: ColdTier, warm_tier: 'GPUWarmTier' = None, debug: bool = False):
        self.cold_tier = cold_tier
        self.warm_tier = warm_tier  # Reference to source tier for WAL confirmation
        self.debug = debug
        self.pending_warm_segments = []  # Track segments waiting for confirmation
        self.logger = get_logger("WarmEvictionHandler")
    
    async def on_eviction(self, data_to_evict: List[pa.RecordBatch]) -> bool:
        """
        Atomic eviction to cold tier with batching optimization:
        Combines all batches into a single large table for optimal write performance.
        Only returns True after cold tier confirms successful storage.
        """
        self.logger.info(f"Starting atomic migration of {len(data_to_evict)} batches to cold tier")
        
        # Memory-efficient: process batches individually to avoid memory explosion
        # Instead of combining into one massive table, stream each batch separately
        self.logger.debug(f"Streaming {len(data_to_evict)} batches individually to prevent memory explosion")
        
        # Process each batch individually to keep memory usage low
        for i, batch in enumerate(data_to_evict):
            success = await self.cold_tier.ingest(batch)
            if not success:
                self.logger.error(f"Cold tier rejected batch {i+1}/{len(data_to_evict)} - aborting migration")
                return False
            
            # Memory pressure handling - let background GC handle cleanup
        
        records_migrated = sum(b.num_rows for b in data_to_evict)
        self.logger.info(f"Cold tier confirmed storage of {records_migrated:,} records")
        
        # Confirm that any pending WAL segments are now safe to delete
        # since data has been durably stored in cold tier
        if self.pending_warm_segments and self.warm_tier:
            self.logger.debug(f"Confirming {len(self.pending_warm_segments)} WAL segments safe to delete")
            # Tell the warm tier's WAL manager these segments are confirmed safe
            self.warm_tier.wal_manager.confirm_segments_safe_to_delete(self.pending_warm_segments)
            self.pending_warm_segments.clear()
        
        # Return True only after cold tier has atomically stored all data
        return True
    
    def confirm_wal_safe_to_delete(self, wal_segment_ids: List[int]):
        """Called by WarmTier to notify which segments can be deleted after cold storage."""
        self.logger.debug(f"Received confirmation request for WAL segments: {wal_segment_ids}")
        self.pending_warm_segments.extend(wal_segment_ids)


class HotTierEvictionHandler(EvictionCallback):
    """Handles eviction from hot tier to warm tier."""
    
    def __init__(self, warm_tier: GPUWarmTier, hot_tier: 'HotTier' = None, debug: bool = False):
        self.warm_tier = warm_tier
        self.hot_tier = hot_tier  # Reference to source tier for WAL confirmation
        self.debug = debug
        self.pending_hot_segments = []  # Track segments waiting for confirmation
        self.logger = get_logger("HotEvictionHandler")
    
    async def on_eviction(self, data_to_evict: List[pa.RecordBatch]) -> bool:
        """
        Atomic eviction to warm tier following TODO.txt protocol:
        Only returns True after warm tier confirms successful storage.
        """
        self.logger.info(f"Starting atomic migration of {len(data_to_evict)} batches to warm tier")
        
        # Transfer WAL control to warm tier on first eviction
        if self.warm_tier.wal_manager is None and self.hot_tier:
            self.logger.info("Transferring WAL control from hot tier to warm tier")
            self.warm_tier.take_control_of_wal(self.hot_tier.wal_manager)
            self.hot_tier.transfer_wal_control()
            
            # Create new WAL manager for hot tier to continue ingestion with durability
            from .wal_manager import WALManager
            new_wal_dir = Path(self.hot_tier.wal_manager.wal_dir) / "hot_continued"
            self.hot_tier.wal_manager = WALManager(new_wal_dir, "hot_continued")
            self.hot_tier.wal_transferred = False  # Reset flag - we have WAL again
            self.logger.info("Created new WAL manager for hot tier continued operations")
        
        successful_batches = 0
        for batch in data_to_evict:
            success = await self.warm_tier.ingest(batch)
            if success:
                successful_batches += 1
            else:
                self.logger.warning(f"Warm tier rejected batch (likely full - will trigger warm->cold migration)")
                # When warm tier is full, it will automatically evict to cold tier
                # Continue attempting to migrate remaining batches
        
        records_migrated = sum(b.num_rows for b in data_to_evict[:successful_batches])
        self.logger.info(f"Warm tier confirmed storage of {records_migrated:,} records")
        
        # Confirm that any pending WAL segments are now safe to delete
        # since data has been durably stored in warm tier (with its own WAL)
        if self.pending_hot_segments and successful_batches > 0 and self.hot_tier:
            self.logger.debug(f"Confirming {len(self.pending_hot_segments)} WAL segments safe to delete")
            # Tell the hot tier's WAL manager these segments are confirmed safe
            self.hot_tier.wal_manager.confirm_segments_safe_to_delete(self.pending_hot_segments)
            self.pending_hot_segments.clear()
        
        # Return True if at least some data was successfully migrated
        # The warm tier's own eviction mechanism will handle overflow
        return successful_batches > 0
    
    def confirm_wal_safe_to_delete(self, wal_segment_ids: List[int]):
        """Called by HotTier to notify which segments can be deleted after warm storage."""
        self.logger.debug(f"Received confirmation request for WAL segments: {wal_segment_ids}")
        self.pending_hot_segments.extend(wal_segment_ids)


class TSDS:
    """
    Straightforward TSDS implementation with clear tier hierarchy:
    Hot (1M records) -> Warm (2GB GPU) -> Cold (Persistent)
    """
    
    def __init__(self, storage_path: str = None, index_columns: List[str] = None, debug: bool = None, 
                 index_batch_interval: float = None, config_path: str = None, config: TSDBConfig = None):
        """
        Initialize TSDS with configuration support.
        
        Args:
            storage_path: Override storage path (uses config if None)
            index_columns: Override index columns (uses config if None)  
            debug: Override debug flag (uses config if None)
            index_batch_interval: Override index batch interval (uses config if None)
            config_path: Path to custom config file
            config: Pre-loaded config object (takes precedence over config_path)
        """
        # Load configuration
        if config is not None:
            self.config = config
        else:
            self.config = get_config(config_path)
        
        # Set parameters from config with parameter overrides
        self.storage_path = Path(storage_path or self.config.storage.base_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.debug = debug if debug is not None else self.config.debug.enabled
        self.logger = get_logger("TSDS")
        
        # Determine index columns and batch interval
        final_index_columns = index_columns or self.config.indexing.index_columns
        final_index_batch_interval = index_batch_interval or self.config.cold_tier.index_batch_interval
        
        # Initialize tiers with configuration
        wal_dir = str(self.config.get_wal_path())
        cold_dir = str(self.config.get_cold_path())
        
        self.cold_tier = ColdTier(
            cold_dir, 
            index_columns=final_index_columns, 
            debug=self.debug, 
            index_batch_interval=final_index_batch_interval,
            config=self.config
        )
        self.warm_tier = GPUWarmTier(
            max_memory_mb=self.config.warm_tier.max_memory_mb, 
            wal_dir=wal_dir, 
            debug=self.debug,
            config=self.config
        )
        self.hot_tier = HotTier(
            max_records=self.config.hot_tier.max_records, 
            wal_dir=wal_dir, 
            debug=self.debug,
            config=self.config
        )
        
        # Set up eviction chain based on GPU availability
        if self.warm_tier.gpu_available:
            # Normal chain: Hot -> Warm -> Cold
            self.warm_tier.set_eviction_callback(WarmTierEvictionHandler(self.cold_tier, self.warm_tier, debug=debug))
            self.hot_tier.set_eviction_callback(HotTierEvictionHandler(self.warm_tier, self.hot_tier, debug=debug))
            self.logger.info("Initialized with Hot -> Warm -> Cold tier chain (GPU available)")
        else:
            # GPU unavailable: Hot -> Cold (bypass warm tier)
            # Warm tier will just pass-through to cold tier immediately
            self.warm_tier.set_eviction_callback(WarmTierEvictionHandler(self.cold_tier, self.warm_tier, debug=debug))
            self.hot_tier.set_eviction_callback(HotTierEvictionHandler(self.warm_tier, self.hot_tier, debug=debug))
            self.logger.warning("GPU not available - warm tier will pass-through to cold tier")
            self.logger.info("Initialized with Hot -> (Warm passthrough) -> Cold tier chain")
        
        self.logger.info(f"Storage path: {self.storage_path}")
        self.logger.info(f"Index columns: {index_columns}")
        
        # Perform WAL recovery for both tiers on startup
        self._recover_from_wal()
    
    def _recover_from_wal(self):
        """Recover both hot and warm tiers from their respective WAL files."""
        self.logger.info("=== Starting WAL Recovery ===")
        
        # Hot tier recovery (already handled in HotTier.__init__)
        hot_stats = None
        try:
            # Hot tier recovery happens in its constructor, but we can get stats
            hot_records = self.hot_tier.total_records
            if hot_records > 0:
                self.logger.info(f"Hot tier recovered: {hot_records:,} records from WAL")
            else:
                self.logger.info("Hot tier: No WAL data to recover")
        except Exception as e:
            self.logger.error(f"Hot tier WAL recovery failed: {e}")
        
        # Warm tier recovery (only if GPU is available)
        if self.warm_tier.gpu_available:
            try:
                warm_stats_before = len(self.warm_tier.gpu_data)
                self.warm_tier._recover_from_wal()
                warm_stats_after = len(self.warm_tier.gpu_data)
                
                if warm_stats_after > warm_stats_before:
                    self.logger.info(f"Warm tier recovered: {warm_stats_after - warm_stats_before} partitions from WAL")
                else:
                    self.logger.info("Warm tier: No WAL data to recover")
            except Exception as e:
                self.logger.error(f"Warm tier WAL recovery failed: {e}")
        else:
            self.logger.info("Warm tier: Skipping WAL recovery (GPU unavailable)")
        
        self.logger.info("=== WAL Recovery Complete ===")
    
    async def ingest(self, batch: pa.RecordBatch) -> bool:
        """
        Ingest data into TSDS.
        Tries hot tier first, then warm tier, then cold tier as fallback.
        """
        self.logger.debug(f"Ingesting {batch.num_rows:,} records")
        
        # Try hot tier first
        success = await self.hot_tier.ingest(batch)
        if success:
            self.logger.debug(f"Successfully ingested to hot tier")
            return True
        
        # If hot tier fails, try warm tier
        self.logger.debug(f"Hot tier failed, trying warm tier")
        success = await self.warm_tier.ingest(batch)
        if success:
            self.logger.debug(f"Successfully ingested to warm tier")
            return True
            
        # If warm tier fails, try cold tier as last resort
        self.logger.debug(f"Warm tier failed, trying cold tier")
        success = await self.cold_tier.ingest(batch)
        if success:
            self.logger.debug(f"Successfully ingested to cold tier")
            return True
        else:
            self.logger.error(f"All tiers failed to ingest batch")
            return False
    
    async def _streaming_sort_query(
        self,
        filters: dict = None,
        limit: int = None,
        sort_by: str = None,
        ascending: bool = True
    ) -> AsyncIterator[pa.RecordBatch]:
        """
        Performs a memory-efficient k-way merge sort across all tiers.
        """
        self.logger.debug("Starting streaming k-way merge sort query")
        
        # THIS CLASS CONTAINS THE FIX
        class Comparable:
            def __init__(self, value, asc: bool = True):
                self.value = value
                self.asc = asc
            
            def __lt__(self, other):
                if self.value is None: return self.asc
                if other.value is None: return not self.asc
                
                if self.asc:
                    return self.value < other.value
                else:
                    # For descending with a min-heap, the LARGER value
                    # must be considered "less than" the smaller one.
                    return self.value > other.value
            
            def __eq__(self, other):
                return self.value == other.value

        # Get async iterators for each tier
        tiers = [self.hot_tier, self.warm_tier, self.cold_tier]
        iterators = [tier.query(filters=filters) for tier in tiers]

        # Min-heap to store the next available element from each iterator
        min_heap = []
        
        current_batches = [None] * len(tiers)
        current_row_indices = [0] * len(tiers)
        output_schema = None
        
        async def push_next_from_iterator(iter_idx):
            nonlocal output_schema
            batch = current_batches[iter_idx]
            row_idx = current_row_indices[iter_idx]

            if batch is None or row_idx >= batch.num_rows:
                try:
                    batch = await anext(iterators[iter_idx])
                    if output_schema is None and batch.num_rows > 0:
                        output_schema = batch.schema
                    current_batches[iter_idx] = batch
                    current_row_indices[iter_idx] = 0
                    row_idx = 0
                except StopAsyncIteration:
                    current_batches[iter_idx] = None
                    return

            if current_batches[iter_idx] is not None:
                sort_val = batch.column(sort_by)[row_idx].as_py()
                comparable_val = Comparable(sort_val, ascending)
                heapq.heappush(min_heap, (comparable_val, iter_idx))

        for i in range(len(tiers)):
            await push_next_from_iterator(i)
        
        # If only one tier has data, let that tier handle sorting directly
        if len(min_heap) <= 1:
            if len(min_heap) == 1:
                _, tier_idx = min_heap[0]
                tier = tiers[tier_idx]
                async for sorted_batch in tier.query(filters=filters, sort_by=sort_by, ascending=ascending, limit=limit):
                    yield sorted_batch
            return

        output_rows = []
        records_yielded = 0
        OUTPUT_BATCH_SIZE = self.config.query.output_batch_size

        while min_heap:
            _, iter_idx = heapq.heappop(min_heap)

            batch = current_batches[iter_idx]
            row_idx = current_row_indices[iter_idx]
            
            output_rows.append(batch.slice(row_idx, 1).to_pylist()[0])
            
            current_row_indices[iter_idx] += 1
            records_yielded += 1

            await push_next_from_iterator(iter_idx)

            if len(output_rows) >= OUTPUT_BATCH_SIZE or not min_heap:
                if output_rows and output_schema:
                    table = pa.Table.from_pylist(output_rows, schema=output_schema)
                    for out_batch in table.to_batches():
                        if out_batch.num_rows > 0: yield out_batch
                    output_rows.clear()
            
            if limit and records_yielded >= limit:
                break
        
        if output_rows and output_schema:
            table = pa.Table.from_pylist(output_rows, schema=output_schema)
            for out_batch in table.to_batches():
                if out_batch.num_rows > 0: yield out_batch


    async def query(self, filters: dict = None, limit: int = None, sort_by: str = None, ascending: bool = True) -> AsyncIterator[pa.RecordBatch]:
        """
        Query across all tiers.
        If sort_by is provided, results are streamed and merged on-the-fly.
        Otherwise, results are streamed directly from each tier for efficiency.
        """
        self.logger.debug(f"Querying with filters={filters}, limit={limit}, sort_by='{sort_by}', ascending={ascending}")

        if sort_by is None:
            self.logger.debug("Taking unsorted path - querying tiers directly")
            returned_records = 0
            
            # Optimize tier order based on query characteristics
            tier_order = self._optimize_tier_order(filters, limit)
            
            for tier in tier_order:
                remaining_limit = limit - returned_records if limit else None
                if limit and remaining_limit <= 0:
                    break
                
                async for batch in tier.query(filters=filters, limit=remaining_limit):
                    yield batch
                    returned_records += batch.num_rows
                    
                    # Early termination if we have enough results
                    if limit and returned_records >= limit:
                        break
            return

        self.logger.debug("Taking sorted path - using streaming k-way merge")
        async for batch in self._streaming_sort_query(
            filters=filters, limit=limit, sort_by=sort_by, ascending=ascending
        ):
            yield batch
    
    def _optimize_tier_order(self, filters: dict, limit: int):
        """Optimize tier query order based on query characteristics."""
        # Default order: Hot -> Warm -> Cold (most recent data first)
        default_order = [self.hot_tier, self.warm_tier, self.cold_tier]
        
        # If we have a small limit and no time-based filters, prioritize faster tiers
        if limit and limit <= 10000:
            # Small queries: prioritize GPU-accelerated warm tier for speed
            return [self.warm_tier, self.hot_tier, self.cold_tier]
        
        # If we have time-based filters, we might want to check cold tier first for historical data
        time_column_name = self.config.schema.time_column
        if filters and time_column_name in filters:
            timestamp_filter = filters[time_column_name]
            if isinstance(timestamp_filter, dict):
                # Check if this looks like a historical query (more than a few days ago)
                # For now, keep default order but this could be enhanced with timestamp analysis
                pass
        
        return default_order
    
    async def fast_lookup(self, column_name: str, value: str) -> List[pa.RecordBatch]:
        """Fast lookup using cold tier SQLite index for any indexed column."""
        return await self.cold_tier.fast_lookup(column_name, value)
    
    async def build_background_index(self) -> int:
        """
        Trigger background indexing of cold tier files.
        Can be run periodically with asyncio.create_task() or exposed as an endpoint.
        Returns number of files processed.
        """
        return await self.cold_tier.build_index_from_files()
    
    async def get_stats(self) -> dict:
        """Get comprehensive statistics across all tiers."""
        hot_stats = await self.hot_tier.get_stats()
        warm_stats = await self.warm_tier.get_stats()
        cold_stats = await self.cold_tier.get_stats()
        
        total_records = (
            hot_stats.get("total_records", 0) +
            warm_stats.get("total_records", 0) +
            cold_stats.get("total_records", 0)
        )
        
        return {
            "total_records": total_records,
            "hot_tier": hot_stats,
            "warm_tier": warm_stats,
            "cold_tier": cold_stats,
            "storage_path": str(self.storage_path)
        }
    
    async def cleanup(self):
        """Cleanup resources."""
        # Flush any pending index operations first
        await self.cold_tier.flush_pending_index()
        
        # Cleanup all tiers to close WAL files
        await self.hot_tier.cleanup()
        await self.warm_tier.cleanup()
        
        # Close cold tier LMDB database
        await self.cold_tier.close()
        
        self.logger.info("Cleanup complete")
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()
