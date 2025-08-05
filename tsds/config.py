"""
TSDS Configuration Management

Provides centralized configuration management for the TSDS system.
Loads settings from JSON files and environment variables.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
import logging


@dataclass
class StorageConfig:
    """Storage-related configuration."""
    base_path: str
    wal_path: str
    cold_path: str
    logs_path: str


@dataclass
class HotTierConfig:
    """Hot tier configuration."""
    max_records: int
    eviction_threshold_pct: float
    wal_enabled: bool


@dataclass
class WarmTierConfig:
    """Warm tier configuration."""
    max_memory_mb: int
    eviction_threshold_pct: float
    wal_enabled: bool


@dataclass
class ColdTierConfig:
    """Cold tier configuration."""
    compression: str
    index_batch_interval: float
    staging_enabled: bool
    max_thread_workers: int


@dataclass
class QueryConfig:
    """Query-related configuration."""
    output_batch_size: int
    max_concurrent_files: int


@dataclass
class IndexingConfig:
    """Indexing configuration."""
    auto_index: bool
    index_columns: list
    batch_size: int


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str
    max_file_size_mb: int
    max_files: int
    format: str


@dataclass
class PerformanceConfig:
    """Performance tuning configuration."""
    chunk_size: int
    parallel_ingestion: bool
    memory_limit_mb: int
    gpu_memory_limit_mb: int


@dataclass
class SchemaConfig:
    """Schema configuration."""
    time_column: str


@dataclass
class DebugConfig:
    """Debug configuration."""
    enabled: bool
    print_tier_stats: bool
    print_query_plans: bool


class TSDBConfig:
    """Main TSDS configuration manager."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to JSON config file. If None, uses default config.
        """
        self.config_path = config_path
        self._config_data = {}
        self._load_config()
        self._create_config_objects()
    
    def _load_config(self):
        """Load configuration from JSON file and environment variables."""
        # Load default config
        default_config_path = Path(__file__).parent.parent / "tsds_config.json"
        if default_config_path.exists():
            with open(default_config_path, 'r') as f:
                self._config_data = json.load(f)
        else:
            raise FileNotFoundError(f"Default config file not found: {default_config_path}")
        
        # Override with custom config if provided
        if self.config_path:
            config_path = Path(self.config_path)
            if config_path.exists():
                with open(config_path, 'r') as f:
                    custom_config = json.load(f)
                    self._merge_configs(self._config_data, custom_config)
            else:
                raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Override with environment variables
        self._load_env_overrides()
    
    def _merge_configs(self, default: dict, custom: dict):
        """Recursively merge custom config into default config."""
        for key, value in custom.items():
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                self._merge_configs(default[key], value)
            else:
                default[key] = value
    
    def _load_env_overrides(self):
        """Load configuration overrides from environment variables."""
        env_mappings = {
            'TSDS_STORAGE_PATH': ('storage', 'base_path'),
            'TSDS_HOT_MAX_RECORDS': ('hot_tier', 'max_records'),
            'TSDS_WARM_MAX_MEMORY_MB': ('warm_tier', 'max_memory_mb'),
            'TSDS_LOG_LEVEL': ('logging', 'level'),
            'TSDS_DEBUG': ('debug', 'enabled'),
            'TSDS_QUERY_BATCH_SIZE': ('query', 'output_batch_size'),
            'TSDS_TIME_COLUMN': ('schema', 'time_column'),
        }
        
        for env_var, (section, key) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Convert string values to appropriate types
                if key in ['max_records', 'max_memory_mb', 'output_batch_size', 'max_files', 'max_file_size_mb']:
                    value = int(value)
                elif key in ['eviction_threshold_pct', 'index_batch_interval']:
                    value = float(value)
                elif key in ['enabled', 'auto_index', 'wal_enabled', 'staging_enabled']:
                    value = value.lower() in ('true', '1', 'yes', 'on')
                
                if section not in self._config_data:
                    self._config_data[section] = {}
                self._config_data[section][key] = value
    
    def _create_config_objects(self):
        """Create typed configuration objects from loaded data."""
        self.storage = StorageConfig(**self._config_data['storage'])
        self.hot_tier = HotTierConfig(**self._config_data['hot_tier'])
        self.warm_tier = WarmTierConfig(**self._config_data['warm_tier'])
        self.cold_tier = ColdTierConfig(**self._config_data['cold_tier'])
        self.query = QueryConfig(**self._config_data['query'])
        self.indexing = IndexingConfig(**self._config_data['indexing'])
        self.schema = SchemaConfig(**self._config_data['schema'])
        self.logging = LoggingConfig(**self._config_data['logging'])
        self.performance = PerformanceConfig(**self._config_data['performance'])
        self.debug = DebugConfig(**self._config_data['debug'])
    
    def get_storage_path(self) -> Path:
        """Get the main storage path as a Path object."""
        return Path(self.storage.base_path)
    
    def get_wal_path(self) -> Path:
        """Get the WAL storage path."""
        return self.get_storage_path() / self.storage.wal_path
    
    def get_cold_path(self) -> Path:
        """Get the cold tier storage path."""
        return self.get_storage_path() / self.storage.cold_path
    
    def get_logs_path(self) -> Path:
        """Get the logs storage path."""
        return self.get_storage_path() / self.storage.logs_path
    
    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary."""
        return self._config_data.copy()
    
    def save_to_file(self, path: str):
        """Save current configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self._config_data, f, indent=2)
    
    def __repr__(self) -> str:
        return f"TSDBConfig(config_path={self.config_path})"


# Global configuration instance
_global_config: Optional[TSDBConfig] = None


def get_config(config_path: Optional[str] = None) -> TSDBConfig:
    """
    Get the global TSDS configuration instance.
    
    Args:
        config_path: Path to config file. Only used on first call.
    
    Returns:
        TSDBConfig instance
    """
    global _global_config
    if _global_config is None:
        _global_config = TSDBConfig(config_path)
    return _global_config


def reset_config():
    """Reset the global configuration (mainly for testing)."""
    global _global_config
    _global_config = None