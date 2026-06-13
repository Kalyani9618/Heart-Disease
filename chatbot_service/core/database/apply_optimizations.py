"""
Database Optimization Integration Script
========================================
Applies all optimizations to the database and integrates
the new query optimizer with the existing chat history system.

Run this script to:
1. Create partitioned tables
2. Create materialized views
3. Add optimized indexes
4. Test the new optimization layer

Usage:
    conda run -n rag_memory python core/database/apply_optimizations.py
"""

import asyncio
import asyncpg
import logging
import sys
import os
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Suppress deprecation warning for config import
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="config")

from core.config.app_config import get_app_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def apply_optimization_schema(conn: asyncpg.Connection) -> dict:
    """
    Apply the optimization schema to the database.
    This creates partitioned tables, materialized views, and indexes.
    """
    results = {
        "partitioned_tables": False,
        "materialized_views": False,
        "indexes": False,
        "settings": False,
        "errors": []
    }
    
    # 1. Create chat_messages_partitioned table
    logger.info("Creating partitioned chat_messages table...")
    try:
        # Check if partitioned table already exists
        exists = await conn.fetchval("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'chat_messages_partitioned'
            )
        """)
        
        if not exists:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_messages_partitioned (
                    id SERIAL,
                    session_id VARCHAR(255) NOT NULL,
                    user_id VARCHAR(255),
                    role VARCHAR(50) NOT NULL,
                    content TEXT NOT NULL,
                    token_count INTEGER DEFAULT 0,
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (id, created_at)
                ) PARTITION BY RANGE (created_at)
            """)
            logger.info("Created chat_messages_partitioned table")
        else:
            logger.info("chat_messages_partitioned already exists")
        
        results["partitioned_tables"] = True
    except Exception as e:
        error_msg = f"Error creating partitioned table: {e}"
        logger.error(error_msg)
        results["errors"].append(error_msg)
    
    # 2. Create partitions for current and next months
    logger.info("Creating monthly partitions...")
    try:
        current_date = datetime.now()
        for month_offset in range(3):  # Current month + 2 future months
            year = current_date.year
            month = current_date.month + month_offset
            if month > 12:
                month -= 12
                year += 1
            
            partition_name = f"chat_messages_y{year}m{month:02d}"
            start_date = f"{year}-{month:02d}-01"
            
            # Calculate end date (first day of next month)
            next_month = month + 1
            next_year = year
            if next_month > 12:
                next_month = 1
                next_year += 1
            end_date = f"{next_year}-{next_month:02d}-01"
            
            # Check if partition exists
            exists = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = $1
                )
            """, partition_name)
            
            if not exists:
                try:
                    await conn.execute(f"""
                        CREATE TABLE IF NOT EXISTS {partition_name}
                        PARTITION OF chat_messages_partitioned
                        FOR VALUES FROM ('{start_date}') TO ('{end_date}')
                    """)
                    logger.info(f"Created partition: {partition_name}")
                except asyncpg.DuplicateTableError:
                    logger.info(f"Partition {partition_name} already exists")
            else:
                logger.info(f"Partition {partition_name} already exists")
                
    except Exception as e:
        error_msg = f"Error creating partitions: {e}"
        logger.error(error_msg)
        results["errors"].append(error_msg)
    
    # 3. Create optimized indexes
    logger.info("Creating optimized indexes...")
    index_commands = [
        ("idx_chat_messages_part_session_created", 
         "chat_messages_partitioned", 
         "(session_id, created_at DESC)"),
        ("idx_chat_messages_part_user_session", 
         "chat_messages_partitioned", 
         "(user_id, session_id)"),
        ("idx_chat_sessions_user_started", 
         "chat_sessions", 
         "(user_id, started_at DESC)"),
        ("idx_chat_messages_created_brin", 
         "chat_messages_partitioned", 
         "(created_at) USING BRIN"),
        ("idx_chat_messages_timestamp", 
         "chat_messages", 
         "(timestamp DESC)"),
        ("idx_chat_messages_session_timestamp", 
         "chat_messages", 
         "(session_id, timestamp DESC)"),
    ]
    
    for idx_name, table_name, idx_def in index_commands:
        try:
            # Check if table exists first
            table_exists = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = $1
                )
            """, table_name)
            
            if not table_exists:
                logger.warning(f"Skipping index {idx_name}: table {table_name} does not exist")
                continue
            
            # Check if index exists
            idx_exists = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM pg_indexes 
                    WHERE indexname = $1
                )
            """, idx_name)
            
            if not idx_exists:
                if "BRIN" in idx_def:
                    await conn.execute(f"""
                        CREATE INDEX IF NOT EXISTS {idx_name} ON {table_name} 
                        USING BRIN (created_at)
                    """)
                else:
                    await conn.execute(f"""
                        CREATE INDEX IF NOT EXISTS {idx_name} ON {table_name} {idx_def}
                    """)
                logger.info(f"Created index: {idx_name}")
            else:
                logger.info(f"Index {idx_name} already exists")
                
        except Exception as e:
            error_msg = f"Error creating index {idx_name}: {e}"
            logger.warning(error_msg)
            results["errors"].append(error_msg)
    
    results["indexes"] = True
    
    # 4. Create materialized view for recent chat summaries
    logger.info("Creating materialized views...")
    try:
        # Check if materialized view exists
        mv_exists = await conn.fetchval("""
            SELECT EXISTS (
                SELECT FROM pg_matviews 
                WHERE matviewname = 'mv_recent_chat_summary'
            )
        """)
        
        if not mv_exists:
            # Check if source table exists
            table_exists = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'chat_messages'
                )
            """)
            
            if table_exists:
                await conn.execute("""
                    CREATE MATERIALIZED VIEW IF NOT EXISTS mv_recent_chat_summary AS
                    SELECT 
                        session_id,
                        COUNT(*) as message_count,
                        MAX(timestamp) as last_activity,
                        SUM(CASE WHEN message_type = 'human' THEN 1 ELSE 0 END) as user_messages,
                        SUM(CASE WHEN message_type = 'ai' THEN 1 ELSE 0 END) as assistant_messages
                    FROM chat_messages
                    WHERE timestamp > NOW() - INTERVAL '7 days'
                    GROUP BY session_id
                    WITH DATA
                """)
                
                # Create unique index for concurrent refresh
                await conn.execute("""
                    CREATE UNIQUE INDEX IF NOT EXISTS mv_recent_chat_summary_session_idx 
                    ON mv_recent_chat_summary (session_id)
                """)
                logger.info("Created materialized view: mv_recent_chat_summary")
            else:
                logger.warning("Skipping materialized view: chat_messages table does not exist")
        else:
            logger.info("Materialized view mv_recent_chat_summary already exists")
            
        results["materialized_views"] = True
    except Exception as e:
        error_msg = f"Error creating materialized view: {e}"
        logger.warning(error_msg)
        results["errors"].append(error_msg)
    
    # 5. Update PostgreSQL settings (requires superuser, might fail)
    logger.info("Attempting to update connection pool settings...")
    try:
        # These settings require superuser privileges, so they might fail
        # They're typically set in postgresql.conf instead
        await conn.execute("SET work_mem = '256MB'")
        await conn.execute("SET maintenance_work_mem = '512MB'")
        results["settings"] = True
        logger.info("Updated PostgreSQL session settings")
    except Exception as e:
        logger.warning(f"Could not update PostgreSQL settings (may require superuser): {e}")
        results["errors"].append(f"Settings update skipped: {e}")
    
    return results


async def verify_optimizations(conn: asyncpg.Connection) -> dict:
    """Verify that all optimizations are in place"""
    verification = {
        "partitioned_table_exists": False,
        "partitions_count": 0,
        "materialized_views": [],
        "indexes": [],
        "total_tables": 0
    }
    
    # Check partitioned table
    exists = await conn.fetchval("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_name = 'chat_messages_partitioned'
        )
    """)
    verification["partitioned_table_exists"] = exists
    
    # Count partitions
    partitions = await conn.fetch("""
        SELECT child.relname AS partition_name
        FROM pg_inherits
        JOIN pg_class parent ON pg_inherits.inhparent = parent.oid
        JOIN pg_class child ON pg_inherits.inhrelid = child.oid
        WHERE parent.relname = 'chat_messages_partitioned'
    """)
    verification["partitions_count"] = len(partitions)
    
    # List materialized views
    mvs = await conn.fetch("""
        SELECT matviewname FROM pg_matviews 
        WHERE schemaname = 'public'
    """)
    verification["materialized_views"] = [mv["matviewname"] for mv in mvs]
    
    # List optimization indexes
    indexes = await conn.fetch("""
        SELECT indexname FROM pg_indexes 
        WHERE schemaname = 'public' 
        AND (indexname LIKE '%_part_%' OR indexname LIKE 'mv_%' OR indexname LIKE 'idx_%brin%')
    """)
    verification["indexes"] = [idx["indexname"] for idx in indexes]
    
    # Total tables
    total = await conn.fetchval("""
        SELECT COUNT(*) FROM information_schema.tables 
        WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
    """)
    verification["total_tables"] = total
    
    return verification


async def test_query_optimizer():
    """Test the query optimizer module"""
    logger.info("\nTesting Query Optimizer module...")
    
    try:
        from core.database.query_optimizer import (
            OptimizedChatHistoryQueries,
            BatchInsertManager,
            TieredCache,
            MaterializedViewManager
        )
        logger.info("✓ OptimizedChatHistoryQueries imported successfully")
        logger.info("✓ BatchInsertManager imported successfully")
        logger.info("✓ TieredCache imported successfully")
        logger.info("✓ MaterializedViewManager imported successfully")
        
        # Note: Full testing requires database connection
        # This just verifies the module loads correctly
        return True
    except ImportError as e:
        logger.error(f"✗ Failed to import query optimizer modules: {e}")
        return False


async def test_query_monitor():
    """Test the query monitor module"""
    logger.info("\nTesting Query Monitor module...")
    
    try:
        from core.database.query_monitor import (
            QueryPerformanceMonitor,
            SlowQueryLogger,
            DatabaseHealthChecker,
            QueryTimeoutManager,
            QueryMetrics
        )
        
        # Test SlowQueryLogger
        slow_logger = SlowQueryLogger(slow_query_threshold_ms=50.0)
        logger.info("✓ SlowQueryLogger created")
        
        # Test QueryPerformanceMonitor
        monitor = QueryPerformanceMonitor()
        logger.info("✓ QueryPerformanceMonitor created")
        
        # Test QueryTimeoutManager
        timeout_mgr = QueryTimeoutManager()
        assert timeout_mgr.get_timeout("fast") == 5.0
        assert timeout_mgr.get_timeout("normal") == 30.0
        logger.info("✓ QueryTimeoutManager working correctly")
        
        # Test QueryMetrics
        metrics = QueryMetrics(
            query_hash="test_query",
            query_template="SELECT * FROM test",
            execution_time_ms=75.0,
            rows_affected=10,
            timestamp=datetime.utcnow()
        )
        logger.info("✓ QueryMetrics created")
        
        return True
    except Exception as e:
        logger.error(f"✗ Query monitor test failed: {e}")
        return False


async def main():
    """Main function to apply all optimizations"""
    print("=" * 60)
    print("Database Optimization Integration Script")
    print("=" * 60)
    
    # Get database configuration from app config
    config = get_app_config()
    db_config = config.database
    
    # Build connection parameters directly from config
    user = db_config.user
    password = db_config.password
    host = db_config.host
    port = db_config.port
    database = db_config.database
    
    print(f"\nConnecting to PostgreSQL:")
    print(f"  Host: {host}:{port}")
    print(f"  Database: {database}")
    print(f"  User: {user}")
    print()
    
    try:
        # Connect to database
        conn = await asyncpg.connect(
            user=user,
            password=password,
            host=host,
            port=port,
            database=database
        )
        
        logger.info("Connected to database successfully")
        
        # Apply optimizations
        print("\n" + "-" * 40)
        print("Applying Optimizations...")
        print("-" * 40)
        
        results = await apply_optimization_schema(conn)
        
        print("\nOptimization Results:")
        print(f"  Partitioned Tables: {'✓' if results['partitioned_tables'] else '✗'}")
        print(f"  Materialized Views: {'✓' if results['materialized_views'] else '✗'}")
        print(f"  Indexes: {'✓' if results['indexes'] else '✗'}")
        print(f"  Settings: {'✓' if results['settings'] else '⚠ (may require superuser)'}")
        
        if results["errors"]:
            print(f"\nWarnings/Errors ({len(results['errors'])}):")
            for err in results["errors"]:
                print(f"  - {err}")
        
        # Verify optimizations
        print("\n" + "-" * 40)
        print("Verifying Optimizations...")
        print("-" * 40)
        
        verification = await verify_optimizations(conn)
        
        print(f"\nVerification Results:")
        print(f"  Partitioned Table Exists: {'✓' if verification['partitioned_table_exists'] else '✗'}")
        print(f"  Partitions Created: {verification['partitions_count']}")
        print(f"  Materialized Views: {verification['materialized_views']}")
        print(f"  Optimization Indexes: {verification['indexes']}")
        print(f"  Total Tables: {verification['total_tables']}")
        
        await conn.close()
        
        # Test modules
        print("\n" + "-" * 40)
        print("Testing Optimization Modules...")
        print("-" * 40)
        
        optimizer_ok = await test_query_optimizer()
        monitor_ok = await test_query_monitor()
        
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        
        all_ok = (
            results["partitioned_tables"] and
            results["indexes"] and
            optimizer_ok and
            monitor_ok
        )
        
        if all_ok:
            print("\n✓ All database optimizations applied successfully!")
            print("\nNext steps:")
            print("  1. Update chat_history.py to use QueryOptimizer")
            print("  2. Initialize monitoring in app_lifespan.py")
            print("  3. Run performance tests to verify improvements")
        else:
            print("\n⚠ Some optimizations had issues. Check the logs above.")
        
        return all_ok
        
    except Exception as e:
        logger.error(f"Failed to apply optimizations: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result else 1)
