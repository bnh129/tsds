"""
Simple TSDS coordinator that manages Hot -> Warm -> Cold tier flow.
Clean, straightforward implementation focused on core functionality.
"""

import asyncio
import heapq
from typing import List, AsyncIterator, Optional
from pathlib import Path
from datetime import datetime, timezone, timedelta
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
        total_records = sum(b.num_rows for b in data_to_evict)
        self.logger.info(f"Starting atomic migration of {len(data_to_evict)} batches ({total_records:,} records) to cold tier")
        
        # Memory-efficient: process batches individually to avoid memory explosion
        # Instead of combining into one massive table, stream each batch separately
        self.logger.debug(f"Streaming {len(data_to_evict)} batches individually to prevent memory explosion")
        
        # Process each batch individually to keep memory usage low
        # Use increment_count=False to prevent double-counting during migration
        migrated_records = 0
        for i, batch in enumerate(data_to_evict):
            self.logger.info(f"Attempting to ingest batch {i+1}/{len(data_to_evict)} with {batch.num_rows:,} records to cold tier")
            try:
                success = await self.cold_tier.ingest(batch, increment_count=False)
                if not success:
                    self.logger.error(f"Cold tier rejected batch {i+1}/{len(data_to_evict)} - aborting migration")
                    return False
                else:
                    self.logger.info(f"Successfully ingested batch {i+1}/{len(data_to_evict)} to cold tier")
                    migrated_records += batch.num_rows
            except Exception as e:
                self.logger.error(f"Exception during cold tier ingestion: {e}", exc_info=True)
                return False
            
            # Memory pressure handling - let background GC handle cleanup
        
        records_migrated = migrated_records
        self.logger.info(f"Cold tier confirmed storage of {records_migrated:,} records")
        
        # Now that migration is complete, atomically update counts:
        # 1. Cold tier increments count
        self.cold_tier.increment_record_count(records_migrated)
        self.logger.info(f"Cold tier count incremented by {records_migrated:,} records")
        # 2. Warm tier will decrement when it processes the return value
        
        # Confirm that any pending WAL segments are now safe to delete
        # since data has been durably stored in cold tier
        if self.pending_warm_segments and self.warm_tier:
            self.logger.debug(f"Confirming {len(self.pending_warm_segments)} WAL segments safe to delete")
            # Tell the warm tier's WAL manager these segments are confirmed safe
            self.warm_tier.wal_manager.confirm_segments_safe_to_delete(self.pending_warm_segments)
            self.pending_warm_segments.clear()
        
        # Return the actual number of records successfully migrated
        return records_migrated
    
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
        
        # Throttling for spammy warnings
        self._last_warm_reject_warning_time = 0
    
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
        
        successful_batches = []
        failed_batches = []
        
        for i, batch in enumerate(data_to_evict):
            success = await self.warm_tier.ingest(batch)
            if success:
                successful_batches.append((i, batch))
                self.logger.debug(f"Batch {i+1}/{len(data_to_evict)} successfully migrated to warm tier")
            else:
                failed_batches.append((i, batch))
                # Throttle rejection warnings to avoid spam (max once per 30 seconds)
                import time
                current_time = time.time()
                if current_time - self._last_warm_reject_warning_time > 30:
                    self.logger.warning(f"Warm tier rejected batch {i+1}/{len(data_to_evict)} - may be full")
                    self._last_warm_reject_warning_time = current_time
                else:
                    self.logger.debug(f"Warm tier rejected batch {i+1}/{len(data_to_evict)} (full)")
        
        records_migrated = sum(batch.num_rows for _, batch in successful_batches)
        records_failed = sum(batch.num_rows for _, batch in failed_batches)
        
        self.logger.info(f"Hot->Warm migration: {records_migrated:,} records succeeded, {records_failed:,} records failed")
        
        if failed_batches:
            self.logger.error(f"CRITICAL: {len(failed_batches)} batches ({records_failed:,} records) failed to migrate and will be LOST!")
            # TODO: Implement retry mechanism or fallback storage
        
        # Confirm that any pending WAL segments are now safe to delete
        # since data has been durably stored in warm tier (with its own WAL)
        if self.pending_hot_segments and successful_batches and self.hot_tier:
            self.logger.debug(f"Confirming {len(self.pending_hot_segments)} WAL segments safe to delete")
            # Tell the hot tier's WAL manager these segments are confirmed safe
            self.hot_tier.wal_manager.confirm_segments_safe_to_delete(self.pending_hot_segments)
            self.pending_hot_segments.clear()
        
        # Return the actual count of successfully migrated records
        # Hot tier will use this to decrement its count accurately
        return records_migrated
    
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
        
        # Throttling for spammy tier warnings
        self._last_warm_reject_warning_time = 0
        
        # Global ingestion counter for accurate statistics
        self._total_ingested_records = 0
        
        # Determine index columns and batch interval
        final_index_columns = index_columns or self.config.indexing.index_columns
        final_index_batch_interval = index_batch_interval or self.config.cold_tier.index_batch_interval
        
        # Initialize tiers with configuration
        # Use override storage_path if provided, otherwise fall back to config paths
        wal_dir = str(self.storage_path / "wal")
        cold_dir = str(self.storage_path / "cold")
        
        self.cold_tier = ColdTier(
            cold_dir, 
            index_columns=final_index_columns, 
            debug=self.debug, 
            index_batch_interval=final_index_batch_interval,
            config=self.config
        )
        
        # Use simple GPU warm tier (remove LSM complexity)
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
        
        # Set up complete eviction chain: Hot -> Warm -> Cold
        self.warm_tier.set_eviction_callback(WarmTierEvictionHandler(self.cold_tier, self.warm_tier, debug=debug))
        self.hot_tier.set_eviction_callback(HotTierEvictionHandler(self.warm_tier, self.hot_tier, debug=debug))
        self.logger.info("Initialized complete Hot -> Warm -> Cold tier eviction chain")
        
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
        Always tries hot tier first (which always accepts data and triggers migrations as needed).
        """
        self.logger.debug(f"Ingesting {batch.num_rows:,} records")
        
        # Hot tier always accepts data and handles eviction internally
        success = await self.hot_tier.ingest(batch)
        if success:
            # Track total ingested records for accurate statistics
            self._total_ingested_records += batch.num_rows
            self.logger.debug(f"Successfully ingested to hot tier")
            return True
        
        # Hot tier should never fail now, but if it does, try warm tier as backup
        self.logger.warning(f"Hot tier unexpectedly failed, trying warm tier")
        success = await self.warm_tier.ingest(batch)
        if success:
            self._total_ingested_records += batch.num_rows
            self.logger.debug(f"Successfully ingested to warm tier")
            return True
            
        # Warm tier should also never fail now, but if it does, try cold tier as last resort
        self.logger.warning(f"Warm tier unexpectedly failed, trying cold tier")
        success = await self.cold_tier.ingest(batch)
        if success:
            self._total_ingested_records += batch.num_rows
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

        # Intelligent tier selection - skip cold tier when possible for speed
        tiers = self._select_optimal_tiers(filters, limit)
        
        # Create iterators with proper limit pushdown - these start immediately in parallel
        iterators = [tier.query(filters=filters, sort_by=sort_by, ascending=ascending, limit=limit) for tier in tiers]

        # Optimized batch-level merge for pre-sorted streams with parallel fetching
        # Since each tier returns sorted data, we can merge at batch granularity
        
        # Min-heap to store the next available batch from each iterator  
        min_heap = []
        current_batches = [None] * len(tiers)
        output_schema = None
        
        # Use asyncio.gather to fetch first batch from all tiers in parallel
        async def get_next_batch(iter_idx):
            """Get the next batch from an iterator and extract its first sort value."""
            nonlocal output_schema
            try:
                batch = await anext(iterators[iter_idx])
                if output_schema is None and batch.num_rows > 0:
                    output_schema = batch.schema
                current_batches[iter_idx] = batch
                
                if batch.num_rows > 0:
                    # Get the first sort value from this sorted batch
                    sort_val = batch.column(sort_by)[0].as_py()
                    comparable_val = Comparable(sort_val, ascending)
                    heapq.heappush(min_heap, (comparable_val, iter_idx, batch))
                    
            except StopAsyncIteration:
                current_batches[iter_idx] = None

        # Initialize heap with the first batch from each tier IN PARALLEL
        await asyncio.gather(*[get_next_batch(i) for i in range(len(tiers))], return_exceptions=True)
        
        # If only one tier has data, stream directly from that tier
        if len(min_heap) <= 1:
            if len(min_heap) == 1:
                _, tier_idx, first_batch = min_heap[0]
                yield first_batch
                # Continue streaming from that tier
                async for batch in iterators[tier_idx]:
                    yield batch
                    if limit:
                        limit -= batch.num_rows
                        if limit <= 0:
                            break
            return

        # Limit-aware merge: stop as soon as we have enough records
        total_yielded = 0
        result_batches = []
        batch_records = 0
        OUTPUT_BATCH_SIZE = self.config.query.output_batch_size

        while min_heap and (limit is None or total_yielded < limit):
            # Get the batch with the smallest first element
            _, iter_idx, batch = heapq.heappop(min_heap)
            
            result_batches.append(batch)
            batch_records += batch.num_rows
            
            # Get the next batch from the same iterator
            await get_next_batch(iter_idx)
            
            # Yield when we have enough for a batch OR we've hit the limit
            should_yield = (
                batch_records >= OUTPUT_BATCH_SIZE or 
                not min_heap or 
                (limit and total_yielded + batch_records >= limit)
            )
            
            if should_yield and result_batches:
                # Combine batches and sort the combined result
                combined_table = pa.concat_tables([pa.Table.from_batches([b]) for b in result_batches])
                
                # Sort the combined data
                if sort_by in combined_table.schema.names:
                    order = 'ascending' if ascending else 'descending'
                    combined_table = combined_table.sort_by([(sort_by, order)])
                
                # Apply limit if we're close to the total limit
                if limit and total_yielded + combined_table.num_rows > limit:
                    remaining = limit - total_yielded
                    combined_table = combined_table.slice(0, remaining)
                
                # Yield the sorted batch(es)
                for out_batch in combined_table.to_batches(max_chunksize=OUTPUT_BATCH_SIZE):
                    if out_batch.num_rows > 0:
                        yield out_batch
                        total_yielded += out_batch.num_rows
                        
                        # Early termination if we've reached the limit
                        if limit and total_yielded >= limit:
                            return
                
                result_batches = []
                batch_records = 0

    async def _parallel_unsorted_query(self, tier_order: List, filters: dict, limit: int) -> AsyncIterator[pa.RecordBatch]:
        """
        Query tiers in parallel for unsorted results.
        Yields results as soon as they're available from any tier.
        """
        import asyncio
        
        # Create async queues to collect results from each tier
        tier_queues = [asyncio.Queue() for _ in tier_order]
        active_tiers = set(range(len(tier_order)))
        returned_records = 0
        
        async def query_tier(tier_idx: int, tier, queue: asyncio.Queue):
            """Query a single tier and put results in its queue."""
            try:
                # For parallel queries, each tier gets the full limit since we don't know
                # which tier will return results first
                async for batch in tier.query(filters=filters, limit=limit):
                    await queue.put(('batch', batch))
                # Signal completion
                await queue.put(('done', None))
            except Exception as e:
                self.logger.error(f"Error querying tier {tier_idx}: {e}")
                await queue.put(('error', e))
        
        # Start all tier queries concurrently
        tasks = [
            asyncio.create_task(query_tier(i, tier, tier_queues[i]))
            for i, tier in enumerate(tier_order)
        ]
        
        try:
            # Process results as they arrive from any tier
            while active_tiers and (limit is None or returned_records < limit):
                # Wait for next result from any active tier
                done_queues = []
                
                # Check all active tier queues for available results
                for tier_idx in list(active_tiers):
                    queue = tier_queues[tier_idx]
                    try:
                        # Non-blocking check for available results
                        result_type, data = queue.get_nowait()
                        
                        if result_type == 'batch':
                            yield data
                            returned_records += data.num_rows
                            
                            # Check if we've hit the limit
                            if limit and returned_records >= limit:
                                break
                                
                        elif result_type == 'done':
                            done_queues.append(tier_idx)
                            
                        elif result_type == 'error':
                            self.logger.error(f"Tier {tier_idx} query failed: {data}")
                            done_queues.append(tier_idx)
                            
                    except asyncio.QueueEmpty:
                        continue
                
                # Remove completed tiers
                for tier_idx in done_queues:
                    active_tiers.discard(tier_idx)
                
                # If no results available, wait briefly for any tier to produce results
                if not done_queues and active_tiers:
                    # Wait for at least one queue to have data
                    await asyncio.sleep(0.001)  # Small delay to prevent busy waiting
        
        finally:
            # Cancel any remaining tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for all tasks to complete or be cancelled
            await asyncio.gather(*tasks, return_exceptions=True)

    async def query(self, filters: dict = None, limit: int = None, sort_by: str = None, ascending: bool = True) -> AsyncIterator[pa.RecordBatch]:
        """
        Query across all tiers.
        If sort_by is provided, results are streamed and merged on-the-fly.
        Otherwise, results are streamed directly from each tier for efficiency.
        """
        self.logger.debug(f"Querying with filters={filters}, limit={limit}, sort_by='{sort_by}', ascending={ascending}")

        if sort_by is None:
            self.logger.debug("Taking parallel unsorted path - querying tiers concurrently")
            returned_records = 0
            
            # Optimize tier order based on query characteristics
            tier_order = self._optimize_tier_order(filters, limit)
            
            # For unsorted queries, we can query tiers in parallel and yield results as they come
            # This is especially beneficial when tiers have different access speeds (memory vs GPU vs disk)
            async for batch in self._parallel_unsorted_query(tier_order, filters, limit):
                yield batch
                returned_records += batch.num_rows
                if limit and returned_records >= limit:
                    break
            return

        # Remove GPU LSM complexity - use standard tier querying
        
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
    
    def _select_optimal_tiers(self, filters: dict, limit: int):
        """Intelligently select which tiers to query based on filters and limit."""
        
        # For tiny queries (< 1000 records), be very aggressive about using just hot+warm
        if limit and limit < 1000:
            hot_records = self.hot_tier.total_records
            # Better warm tier estimation based on actual GPU data
            warm_records = sum(len(partition) for partition in self.warm_tier.gpu_data.values()) if hasattr(self.warm_tier, 'gpu_data') else 0
            
            self.logger.debug(f"Tiny query analysis: limit={limit}, hot={hot_records}, warm={warm_records}")
            
            # For very small limits, always try hot+warm first (they should have enough data)
            if limit < 100:
                self.logger.debug(f"ULTRA-AGGRESSIVE: Using hot+warm for tiny query (limit={limit})")
                return [self.hot_tier, self.warm_tier]
            
            # For medium-small limits, use hot+warm if we have reasonable data
            elif hot_records + warm_records >= limit:
                self.logger.debug(f"Using hot+warm for small query (limit={limit}, available={hot_records + warm_records})")
                return [self.hot_tier, self.warm_tier]
        
        # For small queries with recent timestamp filters, skip cold tier
        if limit and limit < 10000 and filters:
            time_column_name = self.config.schema.time_column
            if time_column_name in filters:
                timestamp_filter = filters[time_column_name]
                if isinstance(timestamp_filter, dict):
                    # Check if this is a recent query (within last few hours)
                    now = datetime.now(timezone.utc)
                    recent_threshold = now - timedelta(hours=6)  # Extended to 6 hours
                    
                    # Look for >= filter indicating recent data
                    if ">=" in timestamp_filter:
                        start_time = timestamp_filter[">="]
                        try:
                            if hasattr(start_time, 'timestamp'):
                                # Ensure timezone consistency
                                if start_time.tzinfo is None:
                                    start_time = start_time.replace(tzinfo=timezone.utc)
                                if recent_threshold.tzinfo is None:
                                    recent_threshold = recent_threshold.replace(tzinfo=timezone.utc)
                                
                                if start_time >= recent_threshold:
                                    self.logger.debug(f"Skipping cold tier for recent query (>= {start_time})")
                                    return [self.hot_tier, self.warm_tier]
                        except (TypeError, AttributeError):
                            pass
                    
                    # Also check for range queries that are entirely recent
                    if ">=" in timestamp_filter and "<" in timestamp_filter:
                        start_time = timestamp_filter[">="]
                        end_time = timestamp_filter["<"]
                        try:
                            if hasattr(start_time, 'timestamp') and hasattr(end_time, 'timestamp'):
                                # Ensure timezone consistency
                                if start_time.tzinfo is None:
                                    start_time = start_time.replace(tzinfo=timezone.utc)
                                if end_time.tzinfo is None:
                                    end_time = end_time.replace(tzinfo=timezone.utc)
                                if recent_threshold.tzinfo is None:
                                    recent_threshold = recent_threshold.replace(tzinfo=timezone.utc)
                                
                                if start_time >= recent_threshold and end_time >= recent_threshold:
                                    self.logger.debug(f"Skipping cold tier for recent range query")
                                    return [self.hot_tier, self.warm_tier]
                        except (TypeError, AttributeError):
                            pass
        
        # For highly selective filters, try hot+warm first
        if filters and len(filters) >= 2:
            # Multiple filters might be selective enough to avoid cold tier
            self.logger.debug("Using hot+warm tiers for selective multi-filter query")
            return [self.hot_tier, self.warm_tier]
        
        # Default: use all tiers, but skip cold tier if it has no data
        if self.cold_tier.total_records == 0:
            self.logger.debug(f"Skipping empty cold tier (0 records)")
            return [self.hot_tier, self.warm_tier]
        
        self.logger.debug(f"Using all tiers as fallback")
        return [self.hot_tier, self.warm_tier, self.cold_tier]
    
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
    
    async def wait_for_background_processing(self):
        """Wait for all background processing to complete before queries."""
        self.logger.info("No background processing needed with simple GPU warm tier")
    
    async def get_stats(self) -> dict:
        """Get comprehensive statistics across all tiers."""
        hot_stats = await self.hot_tier.get_stats()
        warm_stats = await self.warm_tier.get_stats()
        cold_stats = await self.cold_tier.get_stats()
        
        # Use accurate ingestion counter instead of summing tier counts
        # This avoids double-counting during tier migrations
        total_records = self._total_ingested_records
        
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
