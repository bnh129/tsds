# TSDS - Time Series Data Store

A GPU-accelerated time series datastore built with Python, PyArrow, and CuPy. TSDS features a three-tier architecture designed for throughput, reliability, and scalability.

## Architecture Overview

```
┌─────────────────┐     ┌─────────────────┐    ┌─────────────────┐
│   Hot Tier      │───▶│   Warm Tier     │───▶│   Cold Tier     │
│                 │     │                 │    │                 │
│ • 1M records    │     │ • 2GB GPU mem   │    │ • Unlimited     │
│ • In-memory     │     │ • CuPy accel    │    │ • Parquet files │
│ • WAL durability│     │ • WAL durability│    │ • Snappy comp   │
│ • 95% eviction  │     │ • 95% eviction  │    │ • Date partition│
└─────────────────┘     └─────────────────┘    └─────────────────┘
```

### Three-Tier Data Flow

1. **Hot Tier**: Recent data stored in-memory with WAL for durability
2. **Warm Tier**: GPU-accelerated queries with CuPy for ultra-fast filtering
3. **Cold Tier**: Compressed Parquet storage with predicate pushdown

Data automatically flows between tiers as capacity thresholds are reached, ensuring optimal performance for both recent and historical data access.

## Key Features

### High Performance
- **GPU Acceleration**: CuPy-powered warm tier for blazing fast queries
- **Streaming Queries**: Memory-efficient processing with configurable batch sizes
- **K-way Merge Sorting**: Cross-tier sorted queries with minimal memory overhead
- **Predicate Pushdown**: Efficient filtering at the storage layer

### Reliability & Durability
- **Write-Ahead Logging**: Arrow IPC format with fsync durability guarantees
- **Atomic Transactions**: Staging area approach for crash-safe operations
- **Two-Phase Commit**: Safe tier-to-tier data migration
- **Crash Recovery**: Automatic WAL replay on system restart

### Configuration Management
- **JSON Configuration**: Centralized settings with sensible defaults
- **Environment Variables**: Runtime overrides for deployment flexibility
- **Hierarchical Loading**: Default → Custom → Environment priority
- **Hot Reloading**: Configuration changes without restart

### Advanced Querying
- **Complex Filters**: Range, equality, and IN operations
- **Cross-Tier Sorting**: Global ordering across all storage tiers
- **Limit Optimization**: Early termination for efficient top-K queries
- **Schema Flexibility**: Support for various data types and structures

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd tsds

# Install dependencies
pip install pyarrow cupy-cuda12x pandas psutil
```

### Basic Usage

```python
import asyncio
from datetime import datetime, timezone
import pyarrow as pa
from tsds import TSDS

async def main():
    # Initialize TSDS with default configuration
    async with TSDS(storage_path="./my_data") as tsds:
        
        # Create sample data
        timestamps = [datetime.now(timezone.utc)]
        data = pa.RecordBatch.from_arrays([
            pa.array(timestamps, type=pa.timestamp('ns', tz='UTC')),
            pa.array(["BTC/USD"]),
            pa.array([50000.0]),
            pa.array([1.5])
        ], names=['timestamp', 'symbol', 'price', 'volume'])
        
        # Ingest data
        success = await tsds.ingest(data)
        print(f"Ingestion successful: {success}")
        
        # Query data
        async for batch in tsds.query(filters={"symbol": "BTC/USD"}):
            print(f"Found {batch.num_rows} records")

# Run the example
asyncio.run(main())
```

### Configuration

Create `tsds_config.json` to customize behavior:

```json
{
  "storage": {
    "base_path": "./tsds_storage",
    "wal_path": "wal",
    "cold_path": "cold",
    "logs_path": "logs"
  },
  "hot_tier": {
    "max_records": 1000000,
    "eviction_threshold_pct": 0.95,
    "wal_enabled": true
  },
  "warm_tier": {
    "max_memory_mb": 2048,
    "eviction_threshold_pct": 0.95,
    "wal_enabled": true
  },
  "cold_tier": {
    "compression": "snappy",
    "staging_enabled": true
  },
  "query": {
    "output_batch_size": 4096,
    "max_concurrent_files": 10
  }
}
```

## 📈 Comprehensive Demo

The project includes a powerful demo system for testing and benchmarking:

```bash
# Quick test with 10K records
python3 tsds_demo.py 10000

# Medium scale test with 1M records  
python3 tsds_demo.py 1000000

# Full scale test with 100M records
python3 tsds_demo.py 100000000

# Custom options
python3 tsds_demo.py 50000 --mode simple --batch-size 2000
python3 tsds_demo.py 1000000 --no-queries  # Skip query benchmarks
```

### Demo Features
- **Scalable Testing**: From 1K to 100M+ records
- **Dual Schema Modes**: Simple (4 columns) or Trading (8 columns)
- **Performance Metrics**: Ingestion and query throughput
- **Real-time Monitoring**: Memory usage, GPU utilization, tier distribution
- **Accurate Statistics**: Shows exact record counts per tier with no double-counting
- **Query Benchmarks**: Various filter and sort combinations
- **Eviction Visualization**: See data migrate between tiers in real-time

### Sample Demo Output
```
📥 INGESTION PHASE
  Batch 1337/4000: 33,425,000 records (33.4%) | 378,266 rec/s 
  | Hot: 950,000 | Warm: 26,800,000 | Cold: 5,675,000 
  | RAM: 483MB | GPU: 1574MB | ETA: 4.5min

✅ INGESTION COMPLETE
   Total: 100,000,000 records (exact match)
   Hot tier: 950,000 records (1.0%)
   Warm tier: 24,550,000 records (24.6%) 
   Cold tier: 74,500,000 records (74.4%)
```

## Query System

### Filter Operations

```python
# Range queries
filters = {
    "timestamp": {
        ">=": datetime(2024, 1, 1),
        "<": datetime(2024, 2, 1)
    }
}

# IN operations
filters = {"symbol": ["BTC/USD", "ETH/USD", "ADA/USD"]}

# Equality filters
filters = {"side": "buy"}

# Combined filters
filters = {
    "symbol": ["BTC/USD", "ETH/USD"],
    "timestamp": {">=": start_time},
    "price": {">": 45000.0}
}
```

### Sorting and Limits

```python
# Sorted queries with limits
async for batch in tsds.query(
    filters={"symbol": "BTC/USD"},
    sort_by="price",
    ascending=False,  # Descending order
    limit=1000
):
    # Process top 1000 highest-priced BTC records
    process_batch(batch)
```

## Architecture Deep Dive

### Hot Tier (`hot_tier.py`)
- **In-Memory Storage**: PyArrow RecordBatches with fast access
- **WAL Durability**: Arrow IPC stream format with fsync guarantees
- **Automatic Eviction**: Triggers at 95% capacity to warm tier
- **Crash Recovery**: Automatic WAL replay on startup

### Warm Tier (GPU-Accelerated)
**Components:**
- `warm_tier.py`: Main coordinator and lifecycle management
- `warm_tier_cache.py`: PyArrow ↔ CuPy conversion and GPU memory management
- `warm_tier_query.py`: GPU-accelerated filtering and query execution
- `warm_tier_eviction.py`: LRU-based eviction with memory pressure handling

**Features:**
- **CuPy Integration**: Seamless GPU acceleration with CPU fallback
- **Dictionary Encoding**: Efficient string storage on GPU
- **Preallocated Arrays**: Fixed-size GPU arrays for optimal performance
- **Memory Management**: Automatic eviction at 95% GPU memory usage
- **Query Optimization**: GPU-parallel filtering with concurrent streams
- **Accurate Statistics**: Tier counts based on actual data, not preallocated sizes

### Cold Tier (`cold_tier.py`)
- **Parquet Storage**: Efficient columnar format with Snappy compression
- **Date Partitioning**: `year=YYYY/month=MM/day=DD` structure
- **Atomic Writes**: Staging directory + atomic move pattern
- **Predicate Pushdown**: Filter pushdown to Parquet files
- **Memory Efficiency**: Chunked processing for large datasets

### Write-Ahead Logging (`wal_manager.py`)
- **Arrow IPC Format**: Efficient serialization with schema preservation
- **Segmented Files**: 100MB segments with automatic rollover
- **Transactional Cleanup**: Three-phase deletion protocol
- **Recovery Support**: Automatic replay on system restart

## 🔧 Configuration Reference

### Environment Variables

```bash
# Override key settings via environment
export TSDS_STORAGE_PATH="/data/tsds"
export TSDS_HOT_MAX_RECORDS=2000000
export TSDS_WARM_MAX_MEMORY_MB=4096
export TSDS_LOG_LEVEL="DEBUG"
export TSDS_QUERY_BATCH_SIZE=8192
```

### Performance Tuning

**For High Ingestion Throughput:**
```json
{
  "hot_tier": {"max_records": 2000000},
  "warm_tier": {"max_memory_mb": 4096},
  "performance": {"chunk_size": 50000}
}
```

**For Query Performance:**
```json
{
  "query": {
    "output_batch_size": 8192,
    "max_concurrent_files": 20
  },
  "cold_tier": {"compression": "lz4"}
}
```

**For Memory-Constrained Environments:**
```json
{
  "hot_tier": {"max_records": 500000},
  "warm_tier": {"max_memory_mb": 1024},
  "performance": {"memory_limit_mb": 4096}
}
```

## 🛠️ Development

### Running Tests

```bash
# Simple functionality test
python3 simple_test.py

# Comprehensive demo
python3 tsds_demo.py 1000000

# Trading schema test
python3 demo_100m.py
```

### Debugging

Enable debug logging in configuration:

```json
{
  "logging": {"level": "DEBUG"},
  "debug": {
    "enabled": true,
    "print_tier_stats": true,
    "print_query_plans": true
  }
}
```

### GPU Requirements

- **NVIDIA GPU** with CUDA support
- **CuPy**: Install appropriate version for your CUDA version
- **Fallback**: Automatically switches to CPU-only mode if GPU unavailable

## 🚀 Recent Improvements

### Statistics Accuracy (v2024.08)
- **Fixed Tier Counting**: All tiers now count actual data instead of maintaining separate counters
- **Eliminated Double-Counting**: Resolved issues where records were counted multiple times during tier migration
- **Preallocated Array Handling**: Warm tier correctly counts actual records vs. preallocated GPU array sizes
- **Consistent Reporting**: Total record counts now accurately reflect ingested data across all tiers

### Eviction System Enhancements
- **Improved GPU Memory Calculation**: Uses actual `array.nbytes` instead of estimated sizes
- **Fixed Async Coroutine Issues**: Resolved RuntimeWarnings with `asyncio.as_completed()`
- **Better Memory Thresholds**: Increased to 95% for more efficient tier utilization
- **Accurate Eviction Sizing**: Eviction manager correctly calculates memory to free

### Query Performance Optimizations
- **GPU Stream Parallelism**: Multiple CUDA streams for concurrent partition processing
- **Optimized Memory Usage**: Better handling of GPU memory limits and fallbacks
- **Streaming Results**: `asyncio.gather()` approach for more reliable async iteration

## 🔒 Production Considerations

### Durability Guarantees
- **WAL with fsync**: All writes are durable before acknowledging
- **Atomic tier migration**: Two-phase commit prevents data loss
- **Crash recovery**: Automatic WAL replay ensures consistency

### Scalability
- **Horizontal scaling**: Each tier can be tuned independently
- **Storage growth**: Cold tier supports unlimited Parquet files

### Monitoring
- **Comprehensive logging**: Tier-specific performance metrics
- **Memory tracking**: Real-time usage monitoring
- **Query profiling**: Performance breakdown by tier


