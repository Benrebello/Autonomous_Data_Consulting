# Smart Cache System Guide

## Overview
The system implements an intelligent dual-TTL caching system (`SmartCache`) that optimizes response times by caching both direct and complete analysis results with adaptive expiration times.

## Cache Architecture

### Dual TTL System
- **Direct Mode Cache**: 5-minute TTL for simple, fast-expiring responses
- **Complete Mode Cache**: 1-hour TTL for comprehensive analysis results

### Key Generation
- Uses MD5 hash of query + context for unique cache keys
- Context-aware caching prevents collisions between similar queries
- Automatic key generation ensures consistency

## Cache Management

### Automatic Features
- **Expiration Cleanup**: Automatic removal of expired entries
- **Memory Efficient**: In-memory storage with minimal overhead
- **Context Awareness**: Considers query complexity and response mode

### Cache Statistics
- Tracks hit rates and performance metrics
- Provides cache utilization statistics
- Monitors cache size and cleanup operations

## Response Mode Integration

### Direct Mode Caching
- Ideal for simple queries: "What are the central tendencies?"
- Fast retrieval of statistical summaries
- Quick expiration prevents stale simple results

### Complete Mode Caching
- Suitable for complex analysis: "Perform complete EDA"
- Longer retention for comprehensive reports
- Maintains detailed analysis results for extended periods

## Performance Benefits

### Response Time Optimization
- Instant retrieval for repeated queries
- Eliminates redundant LLM calls
- Maintains accuracy with appropriate TTL settings

### Resource Efficiency
- Reduces API costs through intelligent caching
- Minimizes computational overhead
- Optimizes memory usage with automatic cleanup

## Cache Monitoring

### Built-in Analytics
- Cache hit/miss ratios
- Response time improvements
- Cache size and utilization metrics

### Maintenance
- Automatic cleanup prevents memory bloat
- Configurable TTL settings
- Easy cache clearing for maintenance

## Configuration

### TTL Settings
```python
# Default configuration
direct_ttl_seconds = 300    # 5 minutes
complete_ttl_seconds = 3600 # 1 hour
```

### Cache Size Management
- Automatic expiration prevents unbounded growth
- Memory-efficient storage design
- Optional size limits for constrained environments

## Best Practices

### When to Use Direct Cache
- Statistical summaries and basic metrics
- Simple data profiling requests
- Frequently repeated simple queries

### When to Use Complete Cache
- Comprehensive EDA reports
- Complex multi-step analyses
- Detailed business intelligence reports

### Cache Invalidation
- Automatic expiration based on TTL
- Context changes trigger new cache entries
- Manual clearing available for maintenance
