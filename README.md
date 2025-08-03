# TSDS - Time Series Database System

A high-performance, GPU-accelerated time series database built with Python, PyArrow, and CuPy. TSDS features a three-tier architecture designed for throughput, reliability, and scalability.

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Hot Tier      │───▶│   Warm Tier     │───▶│   Cold Tier     │
│                 │    │                 │    │                 │
│ • 1M records    │    │ • 2GB GPU mem   │    │ • Unlimited     │
│ • In-memory     │    │ • CuPy accel    │    │ • Parquet files │
│ • WAL durability│    │ • WAL durability│    │ • Snappy comp   │
│ • 90% eviction  │    │ • 80% eviction  │    │ • Date partition│
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Three-Tier Data Flow

1. **Hot Tier**: Recent data stored in-memory with WAL for durability
2. **Warm Tier**: GPU-accelerated queries with CuPy for ultra-fast filtering
3. **Cold Tier**: Compressed Parquet storage with predicate pushdown

Data automatically flows between tiers as capacity thresholds are reached, ensuring optimal performance for both recent and historical data access.

## ✨ Key Features

### 🔥 High Performance
- **GPU Acceleration**: CuPy-powered warm tier for blazing fast queries
- **Streaming Queries**: Memory-efficient processing with configurable batch sizes
- **K-way Merge Sorting**: Cross-tier sorted queries with minimal memory overhead
- **Predicate Pushdown**: Efficient filtering at the storage layer

### 🛡️ Reliability & Durability
- **Write-Ahead Logging**: Arrow IPC format with fsync durability guarantees
- **Atomic Transactions**: Staging area approach for crash-safe operations
- **Two-Phase Commit**: Safe tier-to-tier data migration
- **Crash Recovery**: Automatic WAL replay on system restart

### ⚙️ Configuration Management
- **JSON Configuration**: Centralized settings with sensible defaults
- **Environment Variables**: Runtime overrides for deployment flexibility
- **Hierarchical Loading**: Default → Custom → Environment priority
- **Hot Reloading**: Configuration changes without restart

### 📊 Advanced Querying
- **Complex Filters**: Range, equality, and IN operations
- **Cross-Tier Sorting**: Global ordering across all storage tiers
- **Limit Optimization**: Early termination for efficient top-K queries
- **Schema Flexibility**: Support for various data types and structures

## 🚀 Quick Start

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
    "eviction_threshold_pct": 0.9,
    "wal_enabled": true
  },
  "warm_tier": {
    "max_memory_mb": 2048,
    "eviction_threshold_pct": 0.8,
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
- **Memory Monitoring**: Real-time resource usage
- **Query Benchmarks**: Various filter and sort combinations

## 🔍 Query System

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

## 🏛️ Architecture Deep Dive

### Hot Tier (`hot_tier.py`)
- **In-Memory Storage**: PyArrow RecordBatches with fast access
- **WAL Durability**: Arrow IPC stream format with fsync guarantees
- **Automatic Eviction**: Triggers at 90% capacity to warm tier
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
- **Memory Management**: Automatic eviction at 80% GPU memory usage
- **Query Optimization**: GPU-parallel filtering operations

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

## 📊 Performance Benchmarks

### Ingestion Performance
- **100M records**: 267,811 records/sec sustained
- **Memory efficient**: <600MB RAM for 100M records
- **GPU utilization**: 1.4GB/2GB (70% efficiency)

### Query Performance  
- **Full scan**: 12.2M records/sec (100M records in 8.2s)
- **Filtered queries**: 1.8-2.5M records/sec
- **Range queries**: Sub-second for most workloads
- **Sorted queries**: 1.1-1.3s for 12K results from 100M records

### Tier Distribution (100M Records)
- **Hot Tier**: 775K records (0.8%) - Recent data
- **Warm Tier**: 34.3M records (34.3%) - GPU-accelerated
- **Cold Tier**: 64.9M records (64.9%) - Compressed storage

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

## 🔒 Production Considerations

### Durability Guarantees
- **WAL with fsync**: All writes are durable before acknowledging
- **Atomic tier migration**: Two-phase commit prevents data loss
- **Crash recovery**: Automatic WAL replay ensures consistency

### Scalability
- **Horizontal scaling**: Each tier can be tuned independently
- **Storage growth**: Cold tier supports unlimited Parquet files
- **Memory management**: Automatic eviction prevents OOM conditions

### Monitoring
- **Comprehensive logging**: Tier-specific performance metrics
- **Memory tracking**: Real-time usage monitoring
- **Query profiling**: Performance breakdown by tier

## 📝 License

[Add your license information here]

## 🤝 Contributing

[Add contribution guidelines here]

## 📞 Support

[Add support information here]

---

**TSDS** - Built for speed, designed for scale, engineered for reliability.
