#!/usr/bin/env python3
"""
TSDS Comprehensive Demo
Flexible demonstration of the Time Series Database System with configurable scale.

Usage:
    python tsds_demo.py 1000          # 1K records (quick test)
    python tsds_demo.py 1000000       # 1M records (medium test)  
    python tsds_demo.py 100000000     # 100M records (full scale test)
    python tsds_demo.py 10000 --batch-size 1000    # Custom batch size
    python tsds_demo.py 50000 --mode simple        # Simple schema mode
    python tsds_demo.py 1000000 --no-queries       # Skip query tests
"""

import asyncio
import argparse
import time
import shutil
from pathlib import Path
from datetime import datetime, timezone, timedelta
import pyarrow as pa
import psutil
import gc
import sys

# Import from the tsds package
from tsds.tsds import TSDS
from tsds.logger import TSDBLogger, get_logger
from tsds.config import get_config


def calculate_optimal_batch_size(total_records: int) -> int:
    """Calculate optimal batch size based on total records for memory efficiency."""
    if total_records < 10_000:
        return min(1000, total_records)
    elif total_records < 100_000:
        return 5_000
    elif total_records < 1_000_000:
        return 10_000
    elif total_records < 10_000_000:
        return 20_000
    else:
        return 25_000  # Large scale - prevent memory explosion


def generate_simple_batch(batch_size: int, batch_idx: int, start_time: datetime) -> pa.RecordBatch:
    """Generate simple test data - lighter schema for quick tests."""
    # Time spread for realistic data aging
    batch_start_time = start_time + timedelta(minutes=batch_idx * 10)
    
    # Generate timestamp data
    timestamp_array = pa.array([
        batch_start_time + timedelta(seconds=i) for i in range(batch_size)
    ], type=pa.timestamp('ns', tz='UTC'))

    # Generate symbol data
    symbols = ["TSDS_A", "TSDS_B", "TSDS_C"]
    symbol_cycle = symbols * (batch_size // len(symbols) + 1)
    symbol_array = pa.array(symbol_cycle[:batch_size])

    # Generate price data - unique ascending prices with different ranges per batch
    base_price = 100.0 + batch_idx * 10  # Each batch has different price range
    price_array = pa.array([base_price + i * 0.01 for i in range(batch_size)], type=pa.float64())

    # Generate volume data
    volume_array = pa.array([10 + i for i in range(batch_size)], type=pa.int64())

    schema = pa.schema([
        ('timestamp', pa.timestamp('ns', tz='UTC')),
        ('symbol', pa.string()),
        ('price', pa.float64()),
        ('volume', pa.int64())
    ])

    return pa.RecordBatch.from_arrays([
        timestamp_array, symbol_array, price_array, volume_array
    ], schema=schema)


def generate_trading_batch(batch_size: int, batch_idx: int, start_time: datetime) -> pa.RecordBatch:
    """Generate realistic trading data - comprehensive schema for full-scale tests."""
    # Spread data across multiple days for realistic aging
    days_span = 7
    batch_start_time = start_time - timedelta(days=days_span * (batch_idx / 100))
    
    # Pre-generate reusable arrays for performance
    symbols = ["BTC/USD", "ETH/USD", "ADA/USD", "SOL/USD", "DOT/USD", "LINK/USD"]
    
    # Optimized data generation - minimize string operations
    base_trade_id = f"trade_{batch_idx:04d}_"
    base_user_prefix = "user_"
    base_order_id = f"order_{batch_idx:04d}_"
    
    # Use Arrow arrays directly for better performance
    timestamp_array = pa.array([
        batch_start_time + timedelta(seconds=i) for i in range(batch_size)
    ], type=pa.timestamp('ns', tz='UTC'))
    
    # Pre-compute indices for faster string generation
    trade_ids = [f"{base_trade_id}{i:06d}" for i in range(0, batch_size, max(1, batch_size // 1000))]
    trade_id_array = pa.array(trade_ids * (batch_size // len(trade_ids) + 1))[:batch_size]
    
    # Cycle through symbols efficiently
    symbol_cycle = symbols * (batch_size // len(symbols) + 1)
    symbol_array = pa.array(symbol_cycle[:batch_size])
    
    # Generate user IDs more efficiently
    user_ids = [f"{base_user_prefix}{((batch_idx * 1000 + i) % 10000):06d}" 
                for i in range(0, batch_size, max(1, batch_size // 1000))]
    user_id_array = pa.array(user_ids * (batch_size // len(user_ids) + 1))[:batch_size]
    
    # Generate order IDs efficiently
    order_ids = [f"{base_order_id}{i:06d}" for i in range(0, batch_size, max(1, batch_size // 1000))]
    order_id_array = pa.array(order_ids * (batch_size // len(order_ids) + 1))[:batch_size]
    
    # Use fixed values for numerical data to speed up generation
    price_base = 50000 + (batch_idx % 1000) * 10  # Slight variation per batch
    price_array = pa.array([price_base + (i % 100) for i in range(batch_size)])
    
    volume_array = pa.array([0.1 + (i % 10) * 0.01 for i in range(batch_size)])
    
    # Alternate buy/sell
    side_array = pa.array(['buy' if i % 2 == 0 else 'sell' for i in range(batch_size)])
    
    # Create comprehensive trading schema
    schema = pa.schema([
        ('timestamp', pa.timestamp('ns', tz='UTC')),
        ('trade_id', pa.string()),
        ('symbol', pa.string()),
        ('user_id', pa.string()),
        ('order_id', pa.string()),
        ('price', pa.float64()),
        ('volume', pa.float64()),
        ('side', pa.string())
    ])
    
    return pa.RecordBatch.from_arrays([
        timestamp_array, trade_id_array, symbol_array, user_id_array,
        order_id_array, price_array, volume_array, side_array
    ], schema=schema)


def get_memory_usage():
    """Get current system memory usage."""
    process = psutil.Process()
    memory_info = process.memory_info()
    return {
        'rss_mb': memory_info.rss / (1024 * 1024),
        'vms_mb': memory_info.vms / (1024 * 1024),
        'system_available_mb': psutil.virtual_memory().available / (1024 * 1024)
    }


async def run_comprehensive_queries(tsds, start_time: datetime, schema_mode: str):
    """Run comprehensive query tests based on schema mode."""
    print(f"\n🔍 QUERY PERFORMANCE TESTS")
    print("-" * 40)
    
    if schema_mode == "simple":
        # Simple schema queries
        test_queries = [
            ({"symbol": "TSDS_A"}, 10000, "Symbol filter (TSDS_A)"),
            ({"symbol": ["TSDS_A", "TSDS_B"]}, 15000, "Multi-symbol filter"),
            ({}, None, "Full table scan (no filter)")
        ]
        sort_column = "price"
    else:
        # Trading schema queries
        test_queries = [
            ({"symbol": "BTC/USD"}, 50000, "BTC/USD trades"),
            ({"side": "buy"}, 100000, "Buy orders only"),
            ({"symbol": ["BTC/USD", "ETH/USD"]}, 75000, "BTC/ETH trades"),
            ({}, None, "Full table scan")
        ]
        sort_column = "price"
    
    # Range query setup
    query_start_time = start_time + timedelta(minutes=10)
    query_end_time = start_time + timedelta(minutes=60)
    range_filter = {"timestamp": {">=": query_start_time, "<": query_end_time}}
    
    for filters, limit, description in test_queries:
        print(f"\n  Testing: {description}")
        
        query_start = time.time()
        results = []
        
        async for result_batch in tsds.query(filters=filters, limit=limit):
            results.append(result_batch)
        
        query_time = (time.time() - query_start) * 1000
        total_results = sum(b.num_rows for b in results)
        throughput = total_results / (query_time / 1000) if query_time > 0 else 0
        
        print(f"    Results: {total_results:,} records in {query_time:.1f}ms")
        print(f"    Throughput: {throughput:,.0f} records/sec")
    
    # Sorting tests
    print(f"\n  🔄 SORTING TESTS")
    
    # Descending sort test with limit for performance
    print(f"    Testing query with DESCENDING sort by '{sort_column}' (TOP 10)...")
    query_start = time.time()
    desc_results = []
    async for result_batch in tsds.query(sort_by=sort_column, ascending=False, limit=10):
        desc_results.append(result_batch)
        
    query_time = (time.time() - query_start) * 1000
    total_desc_results = sum(b.num_rows for b in desc_results)
    print(f"      Descending: {total_desc_results:,} records in {query_time:.1f}ms")
    
    # Show sample of descending results
    if desc_results:
        print(f"      Sample descending results (first 5 rows):")
        result_table = pa.Table.from_batches(desc_results)
        sample = result_table.slice(0, 5).to_pandas()
        for _, row in sample.iterrows():
            print(f"        {row['timestamp']} | {row['symbol']} | {sort_column}: {row[sort_column]}")
    
    # Ascending sort test with limit for performance
    print(f"    Testing query with ASCENDING sort by '{sort_column}' (TOP 10)...")
    query_start = time.time()
    asc_results = []
    async for result_batch in tsds.query(sort_by=sort_column, ascending=True, limit=10):
        asc_results.append(result_batch)
        
    query_time = (time.time() - query_start) * 1000
    total_asc_results = sum(b.num_rows for b in asc_results)
    print(f"      Ascending: {total_asc_results:,} records in {query_time:.1f}ms")
    
    # Show sample of ascending results  
    if asc_results:
        print(f"      Sample ascending results (first 5 rows):")
        result_table = pa.Table.from_batches(asc_results)
        sample = result_table.slice(0, 5).to_pandas()
        for _, row in sample.iterrows():
            print(f"        {row['timestamp']} | {row['symbol']} | {sort_column}: {row[sort_column]}")
    
    # Ultra-fast tiny query tests
    print(f"    Testing TINY queries with limits to demonstrate sub-second performance...")
    
    # Test with limit=5 (should trigger ultra-fast path)
    print(f"    Testing tiny query: TOP 5 records with sort...")
    query_start = time.time()
    tiny_results = []
    async for result_batch in tsds.query(sort_by=sort_column, ascending=False, limit=5):
        tiny_results.append(result_batch)
    
    query_time = (time.time() - query_start) * 1000
    total_tiny_results = sum(b.num_rows for b in tiny_results)
    print(f"      TOP 5: {total_tiny_results:,} records in {query_time:.1f}ms")
    
    # Test with limit=50 (should also trigger ultra-fast path)
    print(f"    Testing small query: TOP 50 records with sort...")
    query_start = time.time()
    small_results = []
    async for result_batch in tsds.query(sort_by=sort_column, ascending=True, limit=50):
        small_results.append(result_batch)
    
    query_time = (time.time() - query_start) * 1000
    total_small_results = sum(b.num_rows for b in small_results)
    print(f"      TOP 50: {total_small_results:,} records in {query_time:.1f}ms")


async def demo_tsds(total_records: int, batch_size: int = None, schema_mode: str = "trading", 
                   run_queries: bool = True, storage_dir: str = None):
    """Main TSDS demonstration function."""
    
    # Load configuration
    config = get_config()
    
    # Calculate optimal batch size if not provided
    if batch_size is None:
        batch_size = calculate_optimal_batch_size(total_records)
    
    total_batches = (total_records + batch_size - 1) // batch_size  # Ceiling division
    
    # Setup storage directory
    if storage_dir is None:
        storage_dir = f"./tsds_demo_{total_records//1000}k_storage"
    
    storage_path = Path(storage_dir)
    log_dir = storage_path / "logs"
    
    # Clean previous runs
    if storage_path.exists():
        print(f"🧹 Cleaning previous storage: {storage_path}")
        shutil.rmtree(storage_path)
    
    # Create directories
    storage_path.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    TSDBLogger.setup(log_dir=str(log_dir), log_level=config.logging.level, console_output=True)
    logger = get_logger("TSDBDemo")
    log_file = TSDBLogger.get_log_file()
    
    # Display demo information
    print("🚀 TSDS COMPREHENSIVE DEMO")
    print("=" * 50)
    print(f"📊 Target: {total_records:,} records in {total_batches:,} batches of {batch_size:,}")
    print(f"📁 Storage: {storage_path}")
    print(f"📝 Logs: {log_file}")
    print(f"🗂️  Schema: {schema_mode} mode")
    print(f"📊 Configuration:")
    print(f"   - Hot tier: {config.hot_tier.max_records:,} records")
    print(f"   - Warm tier: {config.warm_tier.max_memory_mb}MB")
    print(f"   - Cold tier: {config.cold_tier.compression} compression")
    print(f"   - Query batch: {config.query.output_batch_size}")
    print()
    
    logger.info(f"=== Starting TSDS Demo: {total_records:,} records ===")
    logger.info(f"Schema mode: {schema_mode}, Batch size: {batch_size}")
    
    # Choose data generator based on schema mode
    generate_batch = generate_simple_batch if schema_mode == "simple" else generate_trading_batch
    
    start_time = datetime.now(timezone.utc)
    
    # Disable indexing for maximum performance on large datasets
    index_columns = None if total_records > 100_000 else []
    
    async with TSDS(storage_path=str(storage_path), index_columns=index_columns, debug=False) as tsds:
        # Ingestion Phase
        print("📥 INGESTION PHASE")
        print("-" * 25)
        
        ingestion_start = time.time()
        total_ingested = 0
        last_report_time = ingestion_start
        
        for batch_idx in range(total_batches):
            batch_start = time.time()
            
            # Generate batch data
            current_batch_size = min(batch_size, total_records - total_ingested)
            batch = generate_batch(current_batch_size, batch_idx, start_time)
            
            # Ingest
            success = await tsds.ingest(batch)
            if success:
                total_ingested += current_batch_size
            else:
                logger.error(f"Failed to ingest batch {batch_idx+1}")
                continue
            
            batch_time = time.time() - batch_start
            current_time = time.time()
            
            # Progress reporting (every 15 seconds or every 500 batches for small datasets)
            report_interval = 15.0 if total_records > 100_000 else 500
            should_report = (current_time - last_report_time >= 15.0) or (batch_idx % 500 == 0) or (batch_idx == total_batches - 1)
            
            if should_report:
                elapsed = current_time - ingestion_start
                overall_throughput = total_ingested / elapsed if elapsed > 0 else 0
                batch_throughput = current_batch_size / batch_time if batch_time > 0 else 0
                
                # Get tier stats
                stats = await tsds.get_stats()
                memory = get_memory_usage()
                
                progress_pct = (total_ingested / total_records) * 100
                eta_seconds = (total_records - total_ingested) / overall_throughput if overall_throughput > 0 else 0
                
                gpu_mem_mb = stats['warm_tier'].get('gpu_memory_mb', 0.0)
                print(f"  Batch {batch_idx+1:4d}/{total_batches}: "
                      f"{total_ingested:8,} records ({progress_pct:5.1f}%) "
                      f"| {batch_throughput:7.0f} rec/s "
                      f"| Hot: {stats['hot_tier']['total_records']:6,} "
                      f"| Warm: {stats['warm_tier']['total_records']:7,} "
                      f"| Cold: {stats['cold_tier']['total_records']:8,} "
                      f"| RAM: {memory['rss_mb']:5.0f}MB "
                      f"| GPU: {gpu_mem_mb:5.0f}MB "
                      f"| ETA: {eta_seconds/60:.1f}min")
                
                last_report_time = current_time
                
                # Periodic garbage collection for large datasets
                if total_records > 1_000_000 and batch_idx % 200 == 0:
                    gc.collect()
        
        # Ingestion summary
        ingestion_time = time.time() - ingestion_start
        overall_throughput = total_ingested / ingestion_time
        
        print(f"\n✅ INGESTION COMPLETE")
        print(f"   Ingested: {total_ingested:,} records in {ingestion_time:.1f}s")
        print(f"   Throughput: {overall_throughput:,.0f} records/sec")
        
        logger.info(f"Ingestion complete: {total_ingested:,} records, {overall_throughput:,.0f} rec/s")
        
        # Final tier statistics
        print(f"\n📈 FINAL TIER DISTRIBUTION")
        print("-" * 35)
        
        final_stats = await tsds.get_stats()
        hot_records = final_stats['hot_tier']['total_records']
        warm_records = final_stats['warm_tier']['total_records'] 
        cold_records = final_stats['cold_tier']['total_records']
        total_stored = hot_records + warm_records + cold_records
        
        print(f"  Hot tier:  {hot_records:8,} records ({hot_records/total_stored*100:5.1f}%)")
        print(f"  Warm tier: {warm_records:8,} records ({warm_records/total_stored*100:5.1f}%)")
        print(f"  Cold tier: {cold_records:8,} records ({cold_records/total_stored*100:5.1f}%)")
        print(f"  Total:     {total_stored:8,} records")
        
        # Memory usage summary
        final_memory = get_memory_usage()
        print(f"\n💾 MEMORY USAGE")
        print("-" * 20)
        print(f"  Process RAM: {final_memory['rss_mb']:.1f}MB")
        print(f"  System available: {final_memory['system_available_mb']:.1f}MB")
        if 'gpu_memory_mb' in final_stats['warm_tier'] and final_stats['warm_tier']['gpu_memory_mb'] > 0:
            print(f"  GPU memory: {final_stats['warm_tier']['gpu_memory_mb']:.1f}MB")
        
        # Wait for background processing before queries
        if run_queries:
            print(f"\n⏳ WAITING FOR BACKGROUND PROCESSING")
            print("-" * 40)
            await tsds.wait_for_background_processing()
            print("✅ Background processing complete!")
            
            await run_comprehensive_queries(tsds, start_time, schema_mode)
        
        logger.info("=== TSDS Demo Completed Successfully ===")
        
        # Final summary
        print(f"\n🎉 DEMO COMPLETED SUCCESSFULLY!")
        print(f"   📊 Processed {total_ingested:,} records")
        print(f"   ⚡ Peak throughput: {overall_throughput:,.0f} records/sec")
        print(f"   🗄️  Data distributed across all tiers")
        print(f"   📁 Storage location: {storage_path}")


def main():
    """Parse arguments and run the demo."""
    parser = argparse.ArgumentParser(
        description="TSDS Comprehensive Demo - Flexible scale testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tsds_demo.py 1000                    # Quick 1K test
  python tsds_demo.py 1000000                 # 1M records 
  python tsds_demo.py 100000000               # Full 100M scale
  python tsds_demo.py 50000 --batch-size 2000 # Custom batch size
  python tsds_demo.py 10000 --mode simple     # Simple schema
  python tsds_demo.py 1000000 --no-queries    # Skip query tests
        """
    )
    
    parser.add_argument("records", type=int, help="Total number of records to generate")
    parser.add_argument("--batch-size", type=int, help="Batch size (auto-calculated if not specified)")
    parser.add_argument("--mode", choices=["simple", "trading"], default="trading", 
                       help="Schema mode: simple (4 columns) or trading (8 columns)")
    parser.add_argument("--no-queries", action="store_true", help="Skip query performance tests")
    parser.add_argument("--storage", type=str, help="Custom storage directory")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.records <= 0:
        print("Error: Number of records must be positive")
        return
    
    if args.batch_size and args.batch_size <= 0:
        print("Error: Batch size must be positive")
        return
    
    # Run the demo
    try:
        asyncio.run(demo_tsds(
            total_records=args.records,
            batch_size=args.batch_size,
            schema_mode=args.mode,
            run_queries=not args.no_queries,
            storage_dir=args.storage
        ))
    except KeyboardInterrupt:
        print("\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()