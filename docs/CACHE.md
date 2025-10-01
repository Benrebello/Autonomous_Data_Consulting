# Plan Cache Guide

## What is cached?
- Successful execution plans can be cached and reused to accelerate similar future runs.

## Cache Key
- For simple_analysis: `("simple_analysis", tool)`
- Otherwise: `(main_intent, first_20_columns_of_default_df)`

## Controls
- Sidebar toggle: "Reuse success plan when available".

## Persistence Options
- Default: in-session only (`st.session_state['plan_cache']`).
- For durability: store to disk (e.g., `./cache/plan_cache.json`) or external backends (S3/GCS/DB/Redis).

## Invalidation
- You may add a UI panel to list and remove entries or clear all.
