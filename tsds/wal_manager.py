"""
WAL (Write-Ahead Log) manager for TSDS tiers using Arrow IPC streams.
Provides durability for hot and warm tier data with mmap and fsync.
"""

import os
import mmap
from pathlib import Path
from typing import List, Optional, Iterator
import pyarrow as pa
import pyarrow.ipc as ipc


class WALSegment:
    """A single WAL segment file using Arrow IPC stream format."""
    
    def __init__(self, file_path: Path, mode: str = 'r+b'):
        self.file_path = file_path
        self.mode = mode
        self.file_handle = None
        self.mmap_handle = None
        self.writer = None
        self.reader = None
        self.is_open = False
        
    def open_for_write(self, schema: pa.Schema):
        """Open WAL segment for writing."""
        # Open file for append
        self.file_handle = open(self.file_path, 'ab')
        
        # Create Arrow IPC writer that appends to file
        self.writer = ipc.new_stream(self.file_handle, schema)
        
        self.is_open = True
    
    def write_batch(self, batch: pa.RecordBatch):
        """Write a batch to the WAL segment and fsync."""
        if not self.is_open or not self.writer:
            raise RuntimeError("WAL segment not open for writing")
        
        # Write batch to Arrow IPC stream
        self.writer.write_batch(batch)
        
        # Force sync to disk (durability guarantee)
        self.file_handle.flush()
        os.fsync(self.file_handle.fileno())
    
    def open_for_read(self):
        """Open WAL segment for reading."""
        if not self.file_path.exists():
            return
        
        self.file_handle = open(self.file_path, 'rb')
        
        # Create Arrow IPC reader
        self.reader = ipc.open_stream(self.file_handle)
        self.is_open = True
    
    def read_batches(self) -> Iterator[pa.RecordBatch]:
        """Read all batches from the WAL segment."""
        if not self.is_open or not self.reader:
            return
        
        try:
            for batch in self.reader:
                yield batch
        except Exception as e:
            # Handle partial/corrupted segments gracefully
            print(f"WAL segment {self.file_path} read error: {e}")
            return
    
    def close(self):
        """Close WAL segment and cleanup resources."""
        if self.writer:
            self.writer.close()
            self.writer = None
        
        if self.reader:
            self.reader.close()
            self.reader = None
        
        
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None
        
        self.is_open = False
    
    def delete(self):
        """Delete the WAL segment file."""
        self.close()
        if self.file_path.exists():
            self.file_path.unlink()


class WALManager:
    """Manages multiple WAL segments for a tier."""
    
    def __init__(self, wal_dir: Path, tier_name: str):
        self.wal_dir = wal_dir
        self.tier_name = tier_name
        self.wal_dir.mkdir(parents=True, exist_ok=True)
        
        self.active_segment: Optional[WALSegment] = None
        self.segment_counter = 0
        self.max_segment_size_mb = 100  # Roll over after 100MB
        
        # Transactional cleanup: track segments pending disposal
        self.segments_pending_disposal: List[int] = []
        self.confirmed_safe_segments: List[int] = []
    
    def _get_segment_path(self, segment_id: int) -> Path:
        """Get path for a WAL segment."""
        return self.wal_dir / f"{self.tier_name}_wal_{segment_id:06d}.arrow"
    
    def _discover_existing_segments(self) -> List[int]:
        """Find existing WAL segments."""
        pattern = f"{self.tier_name}_wal_*.arrow"
        segment_files = list(self.wal_dir.glob(pattern))
        
        segment_ids = []
        for file_path in segment_files:
            try:
                # Extract segment ID from filename
                name_parts = file_path.stem.split('_')
                if len(name_parts) >= 3:
                    segment_id = int(name_parts[-1])
                    segment_ids.append(segment_id)
            except ValueError:
                continue
        
        return sorted(segment_ids)
    
    def create_new_segment(self, schema: pa.Schema):
        """Create a new active WAL segment."""
        if self.active_segment:
            self.active_segment.close()
        
        # Find next segment ID
        existing_segments = self._discover_existing_segments()
        if existing_segments:
            self.segment_counter = max(existing_segments) + 1
        else:
            self.segment_counter = 0
        
        # Create new segment
        segment_path = self._get_segment_path(self.segment_counter)
        self.active_segment = WALSegment(segment_path)
        self.active_segment.open_for_write(schema)
    
    def write_batch(self, batch: pa.RecordBatch):
        """Write batch to active WAL segment."""
        if not self.active_segment:
            self.create_new_segment(batch.schema)
        
        # Check if we need to roll over to new segment
        current_size_mb = self.active_segment.file_path.stat().st_size / (1024 * 1024)
        if current_size_mb > self.max_segment_size_mb:
            self.create_new_segment(batch.schema)
        
        self.active_segment.write_batch(batch)
    
    def recover_all_batches(self) -> Iterator[pa.RecordBatch]:
        """Recover all batches from existing WAL segments."""
        existing_segments = self._discover_existing_segments()
        
        for segment_id in existing_segments:
            segment_path = self._get_segment_path(segment_id)
            segment = WALSegment(segment_path)
            
            try:
                segment.open_for_read()
                for batch in segment.read_batches():
                    yield batch
            finally:
                segment.close()
    
    def mark_segments_for_disposal(self, segment_ids: List[int]):
        """Mark segments as pending disposal (transactional step 1)."""
        for segment_id in segment_ids:
            if segment_id not in self.segments_pending_disposal:
                self.segments_pending_disposal.append(segment_id)
    
    def confirm_segments_safe_to_delete(self, segment_ids: List[int]):
        """Confirm segments are safe to delete (transactional step 2)."""
        for segment_id in segment_ids:
            if segment_id not in self.confirmed_safe_segments:
                self.confirmed_safe_segments.append(segment_id)
    
    def cleanup_confirmed_segments(self):
        """Remove only confirmed safe segments (transactional step 3)."""
        segments_to_remove = [
            seg_id for seg_id in self.segments_pending_disposal 
            if seg_id in self.confirmed_safe_segments
        ]
        
        for segment_id in segments_to_remove:
            segment_path = self._get_segment_path(segment_id)
            segment = WALSegment(segment_path)
            segment.delete()
            
            # Remove from tracking lists
            self.segments_pending_disposal.remove(segment_id)
            self.confirmed_safe_segments.remove(segment_id)
        
        return len(segments_to_remove)
    
    def cleanup_old_segments(self, keep_latest: int = 2):
        """Legacy method - remove old WAL segments, keeping only the latest N."""
        existing_segments = self._discover_existing_segments()
        
        if len(existing_segments) <= keep_latest:
            return
        
        segments_to_remove = existing_segments[:-keep_latest]
        for segment_id in segments_to_remove:
            segment_path = self._get_segment_path(segment_id)
            segment = WALSegment(segment_path)
            segment.delete()
    
    def close(self):
        """Close WAL manager and active segment."""
        if self.active_segment:
            self.active_segment.close()
            self.active_segment = None
    
    def get_evictable_segments(self, keep_latest: int = 1) -> List[int]:
        """Get segments that can be marked for disposal."""
        existing_segments = self._discover_existing_segments()
        
        if len(existing_segments) <= keep_latest:
            return []
        
        return existing_segments[:-keep_latest]
    
    def get_stats(self) -> dict:
        """Get WAL statistics."""
        existing_segments = self._discover_existing_segments()
        total_size_mb = 0
        
        for segment_id in existing_segments:
            segment_path = self._get_segment_path(segment_id)
            if segment_path.exists():
                total_size_mb += segment_path.stat().st_size / (1024 * 1024)
        
        return {
            "segment_count": len(existing_segments),
            "total_size_mb": total_size_mb,
            "active_segment": self.segment_counter if self.active_segment else None,
            "pending_disposal": len(self.segments_pending_disposal),
            "confirmed_safe": len(self.confirmed_safe_segments)
        }