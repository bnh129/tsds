"""
Simple storage tier interfaces for greenfield implementation.
Focus on clarity and direct functionality.
"""

from abc import ABC, abstractmethod
from typing import List, AsyncIterator
import pyarrow as pa


class StorageTier(ABC):
    """Base interface for all storage tiers."""
    
    @abstractmethod
    async def ingest(self, batch: pa.RecordBatch) -> bool:
        """
        Ingest a batch of data.
        Returns True if accepted, False if tier is full/cannot accept.
        """
        pass
    
    @abstractmethod
    async def query(self, filters: dict = None, limit: int = None, sort_by: str = None, ascending: bool = True) -> AsyncIterator[pa.RecordBatch]:
        """
        Query data from this tier.
        Tier-level queries are not required to implement sorting, as the top-level
        TSDS coordinator will handle global sorting if requested.
        """
        pass
    
    @abstractmethod
    async def get_stats(self) -> dict:
        """Get tier statistics (record count, memory usage, etc.)."""
        pass


class EvictionCallback:
    """Called when a tier needs to evict data."""
    
    @abstractmethod
    async def on_eviction(self, data_to_evict: List[pa.RecordBatch]) -> bool:
        """
        Handle evicted data (typically by moving to next tier).
        Returns True if eviction was successful.
        """
        pass
    
    def confirm_wal_safe_to_delete(self, wal_segment_ids: List[int]):
        """
        Confirm that the specified WAL segments are safe to delete
        because data has been durably stored in the next tier.
        Default implementation does nothing (for backward compatibility).
        """
        pass

