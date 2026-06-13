"""Text-to-SQL Tool - Phase 1.2 Implementation

Security features:
- Row-level security enforcement via WHERE clause validation
- SQL parsing to verify generated queries
- Database-level read-only permissions recommended
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

try:
    import sqlparse
except ImportError:
    sqlparse = None


from core.prompts.registry import get_prompt

logger = logging.getLogger(__name__)


@dataclass
class SQLQueryResult:
    """Result from Text-to-SQL execution."""
    success: bool
    query: str
    result: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None
    reasoning: Optional[str] = None
    execution_time: Optional[float] = None


class TextToSQLTool:
    """Convert natural language health queries to SQL."""
    
    HEALTH_SCHEMA = """
    Table: patient_vitals
      Columns: id (int), user_id (varchar), vital_type (varchar), vital_value (float), unit (varchar), recorded_at (datetime)
    
    Table: medications
      Columns: id (INT), user_id (VARCHAR), drug_name (VARCHAR), dosage (VARCHAR), frequency (VARCHAR),
               start_date (DATE), end_date (DATE), is_active (BOOLEAN)
    """
    
    FEW_SHOT_EXAMPLES = """
    Example 1:
    Query: "What was my average heart rate last week?"
    Reasoning: Need AVG of vital_value for single metric, no GROUP BY needed
    SQL: SELECT AVG(vital_value) as avg_hr FROM patient_vitals 
         WHERE user_id = :user_id AND vital_type = 'heart_rate'
         AND recorded_at > NOW() - INTERVAL '7 days'
    
    Example 2:
    Query: "Show me my vital types and their average values"
    Reasoning: Multiple columns (vital_type) with aggregate (AVG), must GROUP BY non-aggregated columns
    SQL: SELECT vital_type, AVG(vital_value) as avg_value, COUNT(*) as readings
         FROM patient_vitals 
         WHERE user_id = :user_id
         GROUP BY vital_type
    
    Example 3:
    Query: "What medications am I currently taking?"
    Reasoning: Active medications only (is_active = true or end_date IS NULL)
    SQL: SELECT drug_name, dosage, frequency FROM medications
         WHERE user_id = :user_id AND is_active = true
    
    Example 4:
    Query: "Show me my heart rate readings by day for the past month"
    Reasoning: Multiple rows with grouping by date, must GROUP BY date
    SQL: SELECT DATE(recorded_at) as day, AVG(vital_value) as avg_hr, COUNT(*) as count
         FROM patient_vitals 
         WHERE user_id = :user_id AND vital_type = 'heart_rate'
         AND recorded_at > NOW() - INTERVAL '30 days'
         GROUP BY DATE(recorded_at)
    """
    
    def __init__(self, db=None, llm_gateway=None, db_provider=None):
        """Initialize with database and LLM.
        
        Args:
            db: Database service instance (direct)
            llm_gateway: LLM service instance
            db_provider: Callable that returns the database service (for lazy loading)
        """
        self.db = db
        self.db_provider = db_provider
        self.llm = llm_gateway
    
    def _get_db(self):
        """Retrieve the database service dynamically (lazy loading support)."""
        if self.db:
            return self.db
        if self.db_provider:
            return self.db_provider()
        return None
    
    async def execute(self, query: str, user_id: str) -> SQLQueryResult:
        """Execute natural language query."""
        db_service = self._get_db()
        
        if db_service is None:
            logger.error("TextToSQLTool: Database service unavailable (Initialization pending).")
            return SQLQueryResult(
                success=False, 
                query=query,
                error="Database service unavailable (Initialization pending). Please try again in a few seconds.",
                reasoning="Database connection not yet initialized"
            )

        try:
            sql_query, reasoning = await self._generate_sql(query, user_id)
            
            is_valid, error_msg = self._validate_sql(sql_query)
            if not is_valid:
                return SQLQueryResult(
                    success=False,
                    query=query,
                    error=f"SQL validation failed: {error_msg}",
                    reasoning=reasoning
                )
            
            import time
            start = time.time()
            
            result = await db_service.execute_select(sql_query, {"user_id": user_id})
            
            execution_time = time.time() - start
            
            return SQLQueryResult(
                success=True,
                query=sql_query,
                result=result,
                reasoning=reasoning,
                execution_time=execution_time
            )
        
        except Exception as e:
            logger.error(f"Text-to-SQL error: {e}")
            return SQLQueryResult(
                success=False,
                query=query,
                error=str(e),
                reasoning="Error during SQL generation or execution"
            )
    
    def _get_system_prompt(self) -> str:
        """Get system prompt from centralized PromptRegistry."""
        return get_prompt("tools", "sql_expert")

    async def _generate_sql(self, query: str, user_id: str) -> tuple[str, str]:
        """Generate SQL from natural language using LLM."""
        prompt = f"""{self._get_system_prompt()}

Examples:
{self.FEW_SHOT_EXAMPLES}

EXECUTION GUIDELINES:
- For date ranges, use PostgreSQL interval syntax: NOW() - INTERVAL '7 days'
- Always include WHERE clause with :user_id filter
- If using GROUP BY, list ALL non-aggregated columns and only them
- Use appropriate aggregate functions (AVG, COUNT, MAX, MIN, SUM)
- Do NOT SELECT columns that are not in GROUP BY unless they are aggregates

User Query: {query}

Output format:
<reasoning>Explain step-by-step why this SQL answers the query, including GROUP BY logic if applicable</reasoning>
<sql>SELECT ... (SQL only, no markdown)</sql>
"""
        
        response = await self.llm.generate(prompt)
        
        reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', response, re.DOTALL)
        sql_match = re.search(r'<sql>(.*?)</sql>', response, re.DOTALL)
        
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
        sql = sql_match.group(1).strip() if sql_match else ""
        
        return sql, reasoning
    
    @staticmethod
    def _validate_sql(sql: str) -> tuple[bool, Optional[str]]:
        """
        Validate SQL to prevent injection and enforce row-level security.
        
        Security checks:
        1. Whitelist to allow only SELECT queries
        2. Enforce WHERE clause exists (prevents data exfiltration)
        3. Verify WHERE clause references :user_id parameter (row-level security)
        4. Reject multiple statements
        5. Parse SQL structure to detect bypasses
        """
        dangerous_keywords = [
            'DROP', 'DELETE', 'TRUNCATE', 'INSERT', 'UPDATE',
            'CREATE', 'ALTER', 'EXEC', 'EXECUTE', 'GRANT', 'REVOKE',
            'UNION', 'EXCEPT', 'INTERSECT'  # Prevent UNION-based injection
        ]
        
        sql_upper = sql.upper()
        
        # 1. Verify query type
        if not sql_upper.strip().startswith('SELECT'):
            return False, "Only SELECT queries are allowed"
        
        # 2. Check for dangerous keywords
        for keyword in dangerous_keywords:
            if f' {keyword} ' in f' {sql_upper} ':
                return False, f"Dangerous keyword '{keyword}' detected"
        
        # 3. Reject multiple statements
        if sql.count(';') > 1 or (';' in sql and not sql.rstrip().endswith(';')):
            return False, "Multiple SQL statements not allowed"
        
        # 4. Enhanced: Parse SQL to validate structure
        try:
            if sqlparse:
                parsed = sqlparse.parse(sql)
                if not parsed:
                    return False, "Invalid SQL syntax"
                
                # Check for WHERE clause
                statement = parsed[0]
                has_where = False
                has_user_id_filter = False
                
                token_list = list(statement.flatten())
                where_index = -1
                
                for idx, token in enumerate(token_list):
                    if token.ttype is None and token.value.upper() == 'WHERE':
                        has_where = True
                        where_index = idx
                    
                    # Check if :user_id parameter appears after WHERE
                    if has_where and ':user_id' in token.value.lower():
                        has_user_id_filter = True
                        break
                
                if not has_where:
                    return False, "WHERE clause is required (row-level security)"
                
                if not has_user_id_filter:
                    return False, "WHERE clause must filter by :user_id (row-level security violation)"
        
        except Exception as e:
            # If sqlparse fails, do basic string check
            if 'WHERE' not in sql_upper:
                return False, "WHERE clause is required (row-level security)"
            if ':user_id' not in sql:
                return False, "Query must filter by :user_id parameter"
            logger.warning(f"sqlparse unavailable, using basic validation: {e}")
        
        # 5. Verify :user_id parameter is present (even in WHERE clause)
        if ':user_id' not in sql and 'user_id' not in sql_upper:
            return False, "Query must use :user_id parameter for row-level security"
        
        return True, None
