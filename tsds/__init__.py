"""
Greenfield TSDS Implementation

A clean, straightforward implementation of a three-tier time series database:
- Hot Tier: In-memory cache (1M records)
- Warm Tier: GPU-accelerated storage (2GB limit)
- Cold Tier: Persistent storage (unlimited)

Data flows: Hot -> Warm -> Cold with automatic eviction when tiers are full.
"""

from .tsds import TSDS
from .interfaces import StorageTier, EvictionCallback
from .hot_tier import HotTier
from .warm_tier import GPUWarmTier
from .cold_tier import ColdTier

__all__ = [
    'TSDS',
    'StorageTier', 
    'EvictionCallback',
    'HotTier',
    'GPUWarmTier', 
    'ColdTier'
]