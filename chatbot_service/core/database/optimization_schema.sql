-- ============================================================================
-- PostgreSQL Optimization Schema for HeartGuard AI
-- ============================================================================
-- Implements ChatGPT-style optimizations:
-- 1. Time-based partitioning for chat_messages (billions of rows)
-- 2. BRIN indexes for time-series data (smaller than B-tree)
-- 3. Partial indexes for hot data
-- 4. GIN indexes for JSONB queries
-- 5. Materialized views for common aggregations
-- 6. Query timeout configuration
-- 7. Autovacuum tuning for write-heavy tables
-- ============================================================================

-- ============================================================================
-- PARTITIONED CHAT MESSAGES TABLE
-- ============================================================================
-- Partitioning by month allows:
-- - Fast queries within time ranges
-- - Easy archival (DROP old partitions)
-- - Parallel query execution
-- - Reduced index sizes per partition

-- Drop old table if converting (CAREFUL - backup first!)
-- DROP TABLE IF EXISTS chat_messages CASCADE;

-- Create partitioned parent table
CREATE TABLE IF NOT EXISTS chat_messages_partitioned (
    id BIGSERIAL,
    session_id VARCHAR(255) NOT NULL,
    user_id VARCHAR(255) NOT NULL DEFAULT 'default',
    role VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    metadata_json JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Partition key must be in primary key for partitioned tables
    PRIMARY KEY (id, created_at)
) PARTITION BY RANGE (created_at);

-- Create monthly partitions (auto-create more as needed)
-- Current month
CREATE TABLE IF NOT EXISTS chat_messages_2026_01 
    PARTITION OF chat_messages_partitioned
    FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');

CREATE TABLE IF NOT EXISTS chat_messages_2026_02 
    PARTITION OF chat_messages_partitioned
    FOR VALUES FROM ('2026-02-01') TO ('2026-03-01');

CREATE TABLE IF NOT EXISTS chat_messages_2026_03 
    PARTITION OF chat_messages_partitioned
    FOR VALUES FROM ('2026-03-01') TO ('2026-04-01');

-- Future partitions (create more as needed)
CREATE TABLE IF NOT EXISTS chat_messages_2026_04 
    PARTITION OF chat_messages_partitioned
    FOR VALUES FROM ('2026-04-01') TO ('2026-05-01');

-- Default partition for any data outside defined ranges
CREATE TABLE IF NOT EXISTS chat_messages_default 
    PARTITION OF chat_messages_partitioned
    DEFAULT;

-- ============================================================================
-- OPTIMIZED INDEXES FOR PARTITIONED TABLE
-- ============================================================================

-- Index on session_id (most common query pattern)
CREATE INDEX IF NOT EXISTS idx_chat_part_session 
    ON chat_messages_partitioned (session_id);

-- Composite index for user+time queries (user's recent chats)
CREATE INDEX IF NOT EXISTS idx_chat_part_user_time 
    ON chat_messages_partitioned (user_id, created_at DESC);

-- BRIN index for time-series queries (very small, efficient for sorted data)
CREATE INDEX IF NOT EXISTS idx_chat_part_time_brin 
    ON chat_messages_partitioned USING BRIN (created_at)
    WITH (pages_per_range = 128);

-- GIN index for JSONB metadata queries
CREATE INDEX IF NOT EXISTS idx_chat_part_metadata_gin 
    ON chat_messages_partitioned USING GIN (metadata_json);

-- Partial index for recent data (last 24 hours - hot data)
CREATE INDEX IF NOT EXISTS idx_chat_part_recent 
    ON chat_messages_partitioned (session_id, created_at DESC)
    WHERE created_at > NOW() - INTERVAL '24 hours';


-- ============================================================================
-- OPTIMIZED INDEXES FOR EXISTING TABLES
-- ============================================================================

-- Chat Sessions: Composite index for user lookup with activity filter
CREATE INDEX IF NOT EXISTS idx_sessions_user_activity 
    ON chat_sessions (user_id, last_activity DESC)
    WHERE last_activity > NOW() - INTERVAL '7 days';

-- Chat History (non-partitioned): Session + time for range queries
CREATE INDEX IF NOT EXISTS idx_chat_history_session_time 
    ON chat_history (session_id, timestamp DESC);

-- User Preferences: Fast lookup by category
CREATE INDEX IF NOT EXISTS idx_user_prefs_category 
    ON user_preferences (user_id, category);

-- Vitals: BRIN for time-series (very efficient for sorted insert order)
CREATE INDEX IF NOT EXISTS idx_vitals_time_brin 
    ON vitals USING BRIN (recorded_at)
    WITH (pages_per_range = 128);

-- Vitals: Composite for user+metric queries
CREATE INDEX IF NOT EXISTS idx_vitals_user_metric_time 
    ON vitals (user_id, metric_type, recorded_at DESC);

-- Feedback: Rating analysis index
CREATE INDEX IF NOT EXISTS idx_feedback_rating_type 
    ON feedback (feedback_type, rating)
    WHERE rating IS NOT NULL;

-- Health Alerts: Active alerts index
CREATE INDEX IF NOT EXISTS idx_alerts_active 
    ON health_alerts (user_id, created_at DESC)
    WHERE is_resolved = FALSE;

-- Memory tables: Importance-based retrieval
CREATE INDEX IF NOT EXISTS idx_memori_ltm_importance_desc 
    ON memori_long_term_memory (user_id, importance_score DESC)
    WHERE importance_score > 0.7;

CREATE INDEX IF NOT EXISTS idx_memori_stm_expires_active 
    ON memori_short_term_memory (user_id, expires_at)
    WHERE expires_at > NOW() OR expires_at IS NULL;


-- ============================================================================
-- MATERIALIZED VIEWS FOR COMMON QUERIES
-- ============================================================================

-- Recent messages per session (cached for fast retrieval)
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_recent_messages AS
SELECT 
    session_id,
    user_id,
    role,
    content,
    metadata_json,
    created_at,
    ROW_NUMBER() OVER (PARTITION BY session_id ORDER BY created_at DESC) as msg_rank
FROM chat_messages
WHERE created_at > NOW() - INTERVAL '24 hours'
WITH DATA;

CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_recent_pk 
    ON mv_recent_messages (session_id, msg_rank);
CREATE INDEX IF NOT EXISTS idx_mv_recent_session 
    ON mv_recent_messages (session_id) WHERE msg_rank <= 10;

-- Session statistics (refreshed periodically)
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_session_stats AS
SELECT 
    session_id,
    user_id,
    COUNT(*) as message_count,
    COUNT(CASE WHEN role = 'user' THEN 1 END) as user_messages,
    COUNT(CASE WHEN role = 'assistant' THEN 1 END) as assistant_messages,
    MIN(created_at) as first_message_at,
    MAX(created_at) as last_message_at,
    EXTRACT(EPOCH FROM (MAX(created_at) - MIN(created_at))) as session_duration_sec
FROM chat_messages
GROUP BY session_id, user_id
WITH DATA;

CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_stats_pk 
    ON mv_session_stats (session_id);
CREATE INDEX IF NOT EXISTS idx_mv_stats_user 
    ON mv_session_stats (user_id);

-- User activity summary (for dashboard)
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_user_activity AS
SELECT 
    user_id,
    DATE_TRUNC('day', created_at) as activity_date,
    COUNT(DISTINCT session_id) as sessions_count,
    COUNT(*) as messages_count,
    MAX(created_at) as last_activity
FROM chat_messages
WHERE created_at > NOW() - INTERVAL '30 days'
GROUP BY user_id, DATE_TRUNC('day', created_at)
WITH DATA;

CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_activity_pk 
    ON mv_user_activity (user_id, activity_date);


-- ============================================================================
-- FUNCTION: Refresh Materialized Views
-- ============================================================================

CREATE OR REPLACE FUNCTION refresh_materialized_views()
RETURNS void AS $$
BEGIN
    -- Refresh concurrently to avoid blocking reads
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_recent_messages;
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_session_stats;
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_user_activity;
    
    RAISE NOTICE 'Materialized views refreshed at %', NOW();
END;
$$ LANGUAGE plpgsql;


-- ============================================================================
-- AUTOVACUUM TUNING FOR WRITE-HEAVY TABLES
-- ============================================================================
-- Chat messages table gets many INSERTs - tune autovacuum accordingly

ALTER TABLE chat_messages SET (
    autovacuum_vacuum_scale_factor = 0.01,     -- Vacuum after 1% bloat (default 20%)
    autovacuum_analyze_scale_factor = 0.005,   -- Analyze after 0.5% change (default 10%)
    autovacuum_vacuum_cost_delay = 2,          -- Speed up vacuum (default 2)
    autovacuum_vacuum_cost_limit = 1000        -- More aggressive (default 200)
);

-- Same for partitioned table
ALTER TABLE chat_messages_partitioned SET (
    autovacuum_vacuum_scale_factor = 0.01,
    autovacuum_analyze_scale_factor = 0.005,
    autovacuum_vacuum_cost_delay = 2,
    autovacuum_vacuum_cost_limit = 1000
);

-- Vitals table also write-heavy
ALTER TABLE vitals SET (
    autovacuum_vacuum_scale_factor = 0.01,
    autovacuum_analyze_scale_factor = 0.005
);


-- ============================================================================
-- QUERY TIMEOUT CONFIGURATION
-- ============================================================================
-- Set default timeouts to prevent runaway queries

-- Session-level default timeout (30 seconds)
-- Apply in application connection: SET statement_timeout = '30s';

-- Create function to apply timeouts
CREATE OR REPLACE FUNCTION set_query_timeout(timeout_ms INTEGER)
RETURNS void AS $$
BEGIN
    EXECUTE format('SET statement_timeout = %s', timeout_ms);
END;
$$ LANGUAGE plpgsql;


-- ============================================================================
-- CONNECTION POOLING OPTIMIZATION
-- ============================================================================
-- Create function to check connection health (for PgBouncer/pgpool)

CREATE OR REPLACE FUNCTION connection_health_check()
RETURNS TABLE (
    status TEXT,
    current_connections INTEGER,
    max_connections INTEGER,
    connection_utilization NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        'healthy'::TEXT as status,
        (SELECT count(*)::INTEGER FROM pg_stat_activity WHERE state = 'active'),
        current_setting('max_connections')::INTEGER,
        ROUND(
            (SELECT count(*) FROM pg_stat_activity)::NUMERIC / 
            current_setting('max_connections')::NUMERIC * 100, 
            2
        );
END;
$$ LANGUAGE plpgsql;


-- ============================================================================
-- BATCH INSERT OPTIMIZATION FUNCTION
-- ============================================================================
-- Function for high-performance batch inserts

CREATE OR REPLACE FUNCTION batch_insert_messages(
    p_messages JSONB
)
RETURNS INTEGER AS $$
DECLARE
    inserted_count INTEGER;
BEGIN
    -- Insert from JSON array
    INSERT INTO chat_messages (session_id, user_id, role, content, metadata_json, created_at)
    SELECT 
        msg->>'session_id',
        COALESCE(msg->>'user_id', 'default'),
        msg->>'role',
        msg->>'content',
        COALESCE((msg->'metadata')::JSONB, '{}'::JSONB),
        COALESCE((msg->>'created_at')::TIMESTAMPTZ, NOW())
    FROM jsonb_array_elements(p_messages) AS msg;
    
    GET DIAGNOSTICS inserted_count = ROW_COUNT;
    
    RETURN inserted_count;
END;
$$ LANGUAGE plpgsql;


-- ============================================================================
-- FUNCTION: Get Recent Messages (Optimized)
-- ============================================================================
-- Uses materialized view when available, falls back to direct query

CREATE OR REPLACE FUNCTION get_recent_messages(
    p_session_id VARCHAR(255),
    p_limit INTEGER DEFAULT 50
)
RETURNS TABLE (
    id BIGINT,
    session_id VARCHAR(255),
    role VARCHAR(50),
    content TEXT,
    metadata_json JSONB,
    created_at TIMESTAMPTZ
) AS $$
BEGIN
    -- Try materialized view first (for last 24 hours)
    IF EXISTS (
        SELECT 1 FROM mv_recent_messages 
        WHERE mv_recent_messages.session_id = p_session_id 
        AND msg_rank <= p_limit
    ) THEN
        RETURN QUERY
        SELECT 
            m.msg_rank::BIGINT as id,  -- Use rank as pseudo-id for MV
            m.session_id,
            m.role::VARCHAR(50),
            m.content,
            m.metadata_json,
            m.created_at
        FROM mv_recent_messages m
        WHERE m.session_id = p_session_id
        AND m.msg_rank <= p_limit
        ORDER BY m.created_at ASC;
    ELSE
        -- Fall back to direct query
        RETURN QUERY
        SELECT 
            cm.id,
            cm.session_id,
            cm.role,
            cm.content,
            cm.metadata_json,
            cm.created_at
        FROM chat_messages cm
        WHERE cm.session_id = p_session_id
        ORDER BY cm.created_at DESC
        LIMIT p_limit;
    END IF;
END;
$$ LANGUAGE plpgsql;


-- ============================================================================
-- PARTITION MAINTENANCE FUNCTION
-- ============================================================================
-- Automatically creates new partitions and drops old ones

CREATE OR REPLACE FUNCTION maintain_chat_partitions(
    p_months_ahead INTEGER DEFAULT 3,
    p_months_retention INTEGER DEFAULT 12
)
RETURNS void AS $$
DECLARE
    partition_date DATE;
    partition_name TEXT;
    partition_start DATE;
    partition_end DATE;
BEGIN
    -- Create future partitions
    FOR i IN 0..p_months_ahead LOOP
        partition_date := DATE_TRUNC('month', NOW() + (i || ' months')::INTERVAL)::DATE;
        partition_name := 'chat_messages_' || TO_CHAR(partition_date, 'YYYY_MM');
        partition_start := partition_date;
        partition_end := partition_date + INTERVAL '1 month';
        
        -- Check if partition exists
        IF NOT EXISTS (
            SELECT 1 FROM pg_class 
            WHERE relname = partition_name
        ) THEN
            EXECUTE format(
                'CREATE TABLE IF NOT EXISTS %I PARTITION OF chat_messages_partitioned FOR VALUES FROM (%L) TO (%L)',
                partition_name, partition_start, partition_end
            );
            RAISE NOTICE 'Created partition: %', partition_name;
        END IF;
    END LOOP;
    
    -- Optionally archive old partitions (commented for safety)
    -- FOR partition_date IN 
    --     SELECT DATE_TRUNC('month', NOW() - (p_months_retention || ' months')::INTERVAL)::DATE
    -- LOOP
    --     partition_name := 'chat_messages_' || TO_CHAR(partition_date, 'YYYY_MM');
    --     IF EXISTS (SELECT 1 FROM pg_class WHERE relname = partition_name) THEN
    --         EXECUTE format('DROP TABLE IF EXISTS %I', partition_name);
    --         RAISE NOTICE 'Dropped old partition: %', partition_name;
    --     END IF;
    -- END LOOP;
END;
$$ LANGUAGE plpgsql;


-- ============================================================================
-- SCHEDULED MAINTENANCE
-- ============================================================================
-- Create pg_cron jobs if extension is available

DO $$
BEGIN
    -- Check if pg_cron is available
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_cron') THEN
        -- Refresh materialized views every 5 minutes
        PERFORM cron.schedule(
            'refresh_mv_views',
            '*/5 * * * *',
            'SELECT refresh_materialized_views()'
        );
        
        -- Maintain partitions daily
        PERFORM cron.schedule(
            'maintain_partitions',
            '0 2 * * *',
            'SELECT maintain_chat_partitions(3, 12)'
        );
        
        RAISE NOTICE 'pg_cron jobs scheduled';
    ELSE
        RAISE NOTICE 'pg_cron not available - schedule maintenance manually';
    END IF;
EXCEPTION
    WHEN OTHERS THEN
        RAISE NOTICE 'Could not schedule cron jobs: %', SQLERRM;
END $$;


-- ============================================================================
-- MONITORING VIEWS
-- ============================================================================

-- Slow query log view
CREATE OR REPLACE VIEW v_slow_queries AS
SELECT 
    pid,
    usename,
    datname,
    state,
    query,
    EXTRACT(EPOCH FROM (NOW() - query_start)) as duration_sec,
    query_start,
    wait_event_type,
    wait_event
FROM pg_stat_activity
WHERE state = 'active'
AND query NOT LIKE '%pg_stat_activity%'
AND EXTRACT(EPOCH FROM (NOW() - query_start)) > 1
ORDER BY duration_sec DESC;

-- Table size statistics
CREATE OR REPLACE VIEW v_table_sizes AS
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname || '.' || tablename)) as total_size,
    pg_size_pretty(pg_table_size(schemaname || '.' || tablename)) as table_size,
    pg_size_pretty(pg_indexes_size(schemaname || '.' || tablename)) as index_size,
    n_live_tup as row_estimate
FROM pg_stat_user_tables
ORDER BY pg_total_relation_size(schemaname || '.' || tablename) DESC;

-- Index usage statistics
CREATE OR REPLACE VIEW v_index_usage AS
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan as scans,
    idx_tup_read as tuples_read,
    idx_tup_fetch as tuples_fetched,
    pg_size_pretty(pg_relation_size(indexrelid)) as index_size
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;


-- ============================================================================
-- GRANT PERMISSIONS (adjust as needed)
-- ============================================================================

-- Grant execute on functions
GRANT EXECUTE ON FUNCTION refresh_materialized_views() TO PUBLIC;
GRANT EXECUTE ON FUNCTION set_query_timeout(INTEGER) TO PUBLIC;
GRANT EXECUTE ON FUNCTION connection_health_check() TO PUBLIC;
GRANT EXECUTE ON FUNCTION batch_insert_messages(JSONB) TO PUBLIC;
GRANT EXECUTE ON FUNCTION get_recent_messages(VARCHAR, INTEGER) TO PUBLIC;
GRANT EXECUTE ON FUNCTION maintain_chat_partitions(INTEGER, INTEGER) TO PUBLIC;

-- Grant select on monitoring views
GRANT SELECT ON v_slow_queries TO PUBLIC;
GRANT SELECT ON v_table_sizes TO PUBLIC;
GRANT SELECT ON v_index_usage TO PUBLIC;
