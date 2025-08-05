#!/usr/bin/env python3
"""
Record Audit Test - Track Sequential Records Through All Tiers
============================================================

This test creates sequentially numbered records and tracks them through
hot -> warm -> cold tier transitions to identify any data loss or counting issues.

Usage:
    python3 record_audit_test.py
"""

import asyncio
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pyarrow as pa
import shutil
from typing import Set, Dict, List, Tuple

from tsds.tsds import TSDS
from tsds.config import get_config
from tsds.logger import get_logger

class RecordAuditor:
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.logger = get_logger("RecordAuditor")
        self.sent_records: Set[int] = set()
        self.expected_total = 0
        
    def create_sequential_batch(self, start_id: int, batch_size: int, timestamp_base: datetime) -> pa.RecordBatch:
        """Create a batch with sequential record IDs for tracking."""
        record_ids = list(range(start_id, start_id + batch_size))
        self.sent_records.update(record_ids)
        
        # Create timestamps spread over time to trigger tier transitions
        timestamps = [timestamp_base + timedelta(minutes=i) for i in range(batch_size)]
        
        # Simple schema for easier debugging
        data = {
            'record_id': record_ids,  # Sequential ID for tracking
            'timestamp': timestamps,
            'symbol': ['AUDIT'] * batch_size,  # Fixed symbol for easy filtering
            'value': [float(i) for i in record_ids],  # Value = record_id for verification
            'batch_num': [start_id // batch_size] * batch_size  # Which batch this came from
        }
        
        return pa.RecordBatch.from_pydict(data)
    
    async def audit_tier_contents(self, tsds: TSDS) -> Dict[str, Set[int]]:
        """Query each tier to see what records it actually contains."""
        results = {
            'hot': set(),
            'warm': set(), 
            'cold': set()
        }
        
        self.logger.info("🔍 Auditing tier contents...")
        
        # Query hot tier
        try:
            self.logger.info("Querying hot tier...")
            async for batch in tsds.hot_tier.query():
                if batch.num_rows > 0:
                    record_ids = batch.column('record_id').to_pylist()
                    results['hot'].update(record_ids)
            self.logger.info(f"Hot tier contains {len(results['hot'])} records")
        except Exception as e:
            self.logger.error(f"Error querying hot tier: {e}")
        
        # Query warm tier  
        try:
            self.logger.info("Querying warm tier...")
            async for batch in tsds.warm_tier.query():
                if batch.num_rows > 0:
                    record_ids = batch.column('record_id').to_pylist()
                    results['warm'].update(record_ids)
            self.logger.info(f"Warm tier contains {len(results['warm'])} records")
        except Exception as e:
            self.logger.error(f"Error querying warm tier: {e}")
            
        # Query cold tier
        try:
            self.logger.info("Querying cold tier...")
            async for batch in tsds.cold_tier.query():
                if batch.num_rows > 0:
                    record_ids = batch.column('record_id').to_pylist()
                    results['cold'].update(record_ids)
            self.logger.info(f"Cold tier contains {len(results['cold'])} records")
        except Exception as e:
            self.logger.error(f"Error querying cold tier: {e}")
            
        return results
    
    def analyze_results(self, tier_contents: Dict[str, Set[int]]) -> Dict:
        """Analyze the audit results for data integrity issues."""
        all_found = set()
        for tier_records in tier_contents.values():
            all_found.update(tier_records)
            
        missing = self.sent_records - all_found
        duplicated = []
        
        # Check for duplicates across tiers
        for record_id in all_found:
            tiers_containing = [tier for tier, records in tier_contents.items() if record_id in records]
            if len(tiers_containing) > 1:
                duplicated.append((record_id, tiers_containing))
        
        return {
            'sent_count': len(self.sent_records),
            'found_count': len(all_found),
            'missing_count': len(missing),
            'duplicate_count': len(duplicated),
            'missing_records': sorted(list(missing))[:20],  # Show first 20 missing
            'duplicated_records': duplicated[:20],  # Show first 20 duplicated
            'tier_breakdown': {tier: len(records) for tier, records in tier_contents.items()}
        }
    
    def print_audit_report(self, analysis: Dict, tier_stats: Dict):
        """Print comprehensive audit report."""
        print("\n" + "="*80)
        print("📊 RECORD AUDIT REPORT")
        print("="*80)
        
        print(f"📤 Records sent: {analysis['sent_count']:,}")
        print(f"📥 Records found: {analysis['found_count']:,}")
        print(f"❌ Missing records: {analysis['missing_count']:,}")
        print(f"🔄 Duplicated records: {analysis['duplicate_count']:,}")
        
        if analysis['missing_count'] > 0:
            print(f"⚠️  DATA LOSS: {analysis['missing_count']:,} records ({analysis['missing_count']/analysis['sent_count']*100:.1f}%) are missing!")
            print(f"   First few missing IDs: {analysis['missing_records']}")
        
        if analysis['duplicate_count'] > 0:
            print(f"⚠️  DUPLICATION: {analysis['duplicate_count']:,} records found in multiple tiers!")
            for record_id, tiers in analysis['duplicated_records']:
                print(f"   Record {record_id} found in: {', '.join(tiers)}")
        
        print(f"\n📈 TIER BREAKDOWN:")
        for tier, count in analysis['tier_breakdown'].items():
            print(f"   {tier.title()} tier: {count:,} records")
        
        print(f"\n📊 TIER STATISTICS (reported vs actual):")
        for tier in ['hot', 'warm', 'cold']:
            tier_name = f"{tier}_tier" if tier != "cold" else "cold_tier"
            reported = tier_stats.get(tier_name, {}).get('total_records', 0)
            actual = analysis['tier_breakdown'][tier]
            status = "✅" if reported == actual else "❌"
            print(f"   {tier.title()}: reported={reported:,}, actual={actual:,} {status}")
        
        if analysis['missing_count'] == 0 and analysis['duplicate_count'] == 0:
            print(f"\n✅ AUDIT PASSED: All {analysis['sent_count']:,} records accounted for!")
        else:
            print(f"\n❌ AUDIT FAILED: Data integrity issues detected!")


async def run_audit_test():
    """Run the comprehensive record audit test."""
    storage_path = "./audit_test_storage"
    
    # Clean slate
    if Path(storage_path).exists():
        print(f"🧹 Cleaning previous audit storage: {storage_path}")
        shutil.rmtree(storage_path)
    
    auditor = RecordAuditor(storage_path)
    
    # Test parameters - designed to hit all tiers  
    total_records = 100_000_000  # Smaller test to see eviction logs more clearly
    batch_size = 50_000  # Large batches to trigger warm->cold faster
    num_batches = total_records // batch_size
    
    print(f"🚀 STARTING RECORD AUDIT TEST")
    print(f"📊 Target: {total_records:,} sequential records in {num_batches} batches")
    print(f"📁 Storage: {storage_path}")
    print(f"🎯 Goal: Track every record through all tier transitions")
    
    timestamp_base = datetime.now(timezone.utc) - timedelta(days=30)  # Start 30 days ago
    
    async with TSDS(storage_path=storage_path, debug=False) as tsds:
        print(f"\n📥 INGESTION PHASE")
        print("-" * 40)
        
        start_time = time.time()
        
        for batch_num in range(num_batches):
            start_id = batch_num * batch_size
            batch_timestamp = timestamp_base + timedelta(hours=batch_num)
            
            batch = auditor.create_sequential_batch(start_id, batch_size, batch_timestamp)
            
            success = await tsds.ingest(batch)
            if not success:
                print(f"❌ Batch {batch_num} ingestion failed!")
                return
            
            # Track any ingestion failures
            if not success:
                auditor.logger.error(f"Batch {batch_num} (records {start_id}-{start_id+batch_size-1}) ingestion failed")
            
            # Progress reporting
            if (batch_num + 1) % 10 == 0 or batch_num == num_batches - 1:
                elapsed = time.time() - start_time
                records_so_far = (batch_num + 1) * batch_size
                rate = records_so_far / elapsed if elapsed > 0 else 0
                
                # Get current tier stats
                hot_stats = await tsds.hot_tier.get_stats()
                warm_stats = await tsds.warm_tier.get_stats()
                cold_stats = await tsds.cold_tier.get_stats()
                
                print(f"  Batch {batch_num+1:3d}/{num_batches}: {records_so_far:8,} records | "
                      f"{rate:6.0f} rec/s | "
                      f"Hot: {hot_stats['total_records']:7,} | "
                      f"Warm: {warm_stats['total_records']:8,} | "
                      f"Cold: {cold_stats['total_records']:8,}")
        
        print(f"\n✅ INGESTION COMPLETE")
        elapsed = time.time() - start_time
        print(f"   Ingested: {total_records:,} records in {elapsed:.1f}s")
        print(f"   Throughput: {total_records/elapsed:.0f} records/sec")
        
        # Wait a moment for any pending evictions
        print(f"\n⏳ Waiting for tier transitions to complete...")
        await asyncio.sleep(2)
        
        # Get final tier statistics
        print(f"\n📊 FINAL TIER STATISTICS")
        print("-" * 40)
        
        hot_stats = await tsds.hot_tier.get_stats()
        warm_stats = await tsds.warm_tier.get_stats()
        cold_stats = await tsds.cold_tier.get_stats()
        
        tier_stats = {
            'hot_tier': hot_stats,
            'warm_tier': warm_stats, 
            'cold_tier': cold_stats
        }
        
        total_reported = hot_stats['total_records'] + warm_stats['total_records'] + cold_stats['total_records']
        
        print(f"  Hot tier:   {hot_stats['total_records']:8,} records")
        print(f"  Warm tier:  {warm_stats['total_records']:8,} records")
        print(f"  Cold tier:  {cold_stats['total_records']:8,} records")
        print(f"  Total:      {total_reported:8,} records")
        print(f"  Expected:   {total_records:8,} records")
        
        if total_reported != total_records:
            print(f"  ⚠️  MISMATCH: {total_records - total_reported:+,} records!")
        
        # Now audit actual tier contents
        print(f"\n🔍 AUDITING ACTUAL TIER CONTENTS")
        print("-" * 40)
        
        tier_contents = await auditor.audit_tier_contents(tsds)
        analysis = auditor.analyze_results(tier_contents)
        
        # Print comprehensive report
        auditor.print_audit_report(analysis, tier_stats)


if __name__ == "__main__":
    asyncio.run(run_audit_test())
