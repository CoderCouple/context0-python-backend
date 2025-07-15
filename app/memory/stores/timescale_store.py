import asyncio
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import asyncpg

from app.memory.models.memory_entry import MemoryEntry
from app.memory.stores.base_store import TimeSeriesStore


class TimescaleTimeSeriesStore(TimeSeriesStore):
    """TimescaleDB implementation of TimeSeriesStore for temporal memory queries"""

    def __init__(self, connection_string: str, table_name: str = "memory_timeseries"):
        """Initialize TimescaleDB connection"""
        self.connection_string = connection_string
        self.table_name = table_name
        self.pool = None

    async def initialize(self):
        """Initialize TimescaleDB connection and setup hypertables"""
        self.pool = await asyncpg.create_pool(self.connection_string)

        # Create tables and hypertables
        await self._create_schema()

        # Test connection
        async with self.pool.acquire() as conn:
            await conn.execute("SELECT 1")

    async def _create_schema(self):
        """Create TimescaleDB schema with hypertables and indexes"""
        async with self.pool.acquire() as conn:
            # Create extension if not exists
            await conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")

            # Create main timeseries table with cross-database references
            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    time TIMESTAMPTZ NOT NULL,
                    memory_id VARCHAR(255) NOT NULL,
                    user_id VARCHAR(255) NOT NULL,
                    session_id VARCHAR(255) NOT NULL,
                    memory_type VARCHAR(50) NOT NULL,
                    scope VARCHAR(255) NOT NULL,
                    input_text TEXT,
                    summary TEXT,
                    tags TEXT[],
                    confidence_score DOUBLE PRECISION,
                    processing_time_ms DOUBLE PRECISION,
                    metadata JSONB,
                    
                    -- Cross-database references
                    vector_store_ref VARCHAR(255),     -- Reference to vector store embedding
                    graph_store_refs TEXT[],           -- Array of graph node IDs
                    doc_store_ref VARCHAR(255),        -- Reference to document store entry
                    audit_store_refs TEXT[],           -- Array of audit log entry IDs
                    
                    -- Temporal extraction fields
                    event_date TIMESTAMPTZ,
                    relative_time VARCHAR(50),
                    time_of_day VARCHAR(20),
                    temporal_context JSONB,
                    temporal_references JSONB,         -- References to other temporal events
                    
                    -- Enhanced relationship tracking
                    entity_count INTEGER DEFAULT 0,
                    concept_count INTEGER DEFAULT 0,
                    relationship_types TEXT[],
                    connected_memories TEXT[],          -- IDs of related memories
                    memory_clusters TEXT[],             -- Cluster IDs this memory belongs to
                    
                    -- Contextual information
                    temporal_significance DOUBLE PRECISION DEFAULT 0.0,  -- How significant this event was temporally
                    cross_reference_count INTEGER DEFAULT 0,             -- Number of cross-references
                    
                    PRIMARY KEY (time, memory_id)
                );
            """
            )

            # Create hypertable (only if not already created)
            try:
                await conn.execute(
                    f"""
                    SELECT create_hypertable('{self.table_name}', 'time', 
                                           chunk_time_interval => INTERVAL '1 day',
                                           if_not_exists => TRUE);
                """
                )
            except Exception as e:
                # Hypertable might already exist
                pass

            # Create indexes for performance
            indexes = [
                f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_user_time ON {self.table_name} (user_id, time DESC);",
                f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_memory_type ON {self.table_name} (memory_type, time DESC);",
                f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_session ON {self.table_name} (session_id, time DESC);",
                f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_tags ON {self.table_name} USING GIN (tags);",
                f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_metadata ON {self.table_name} USING GIN (metadata);",
                f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_event_date ON {self.table_name} (event_date) WHERE event_date IS NOT NULL;",
            ]

            for index_sql in indexes:
                try:
                    await conn.execute(index_sql)
                except Exception:
                    # Index might already exist
                    pass

            # Create materialized views for common aggregations
            await self._create_materialized_views()

    async def _create_materialized_views(self):
        """Create materialized views for fast aggregations"""
        async with self.pool.acquire() as conn:
            # Hourly memory count by user
            await conn.execute(
                f"""
                CREATE MATERIALIZED VIEW IF NOT EXISTS memory_hourly_stats AS
                SELECT 
                    time_bucket('1 hour', time) as hour,
                    user_id,
                    memory_type,
                    COUNT(*) as memory_count,
                    AVG(confidence_score) as avg_confidence,
                    AVG(processing_time_ms) as avg_processing_time
                FROM {self.table_name}
                GROUP BY hour, user_id, memory_type
                ORDER BY hour DESC;
            """
            )

            # Daily memory summary
            await conn.execute(
                f"""
                CREATE MATERIALIZED VIEW IF NOT EXISTS memory_daily_stats AS
                SELECT 
                    time_bucket('1 day', time) as day,
                    user_id,
                    COUNT(*) as total_memories,
                    COUNT(DISTINCT memory_type) as unique_types,
                    COUNT(DISTINCT session_id) as unique_sessions,
                    array_agg(DISTINCT memory_type) as memory_types,
                    AVG(confidence_score) as avg_confidence
                FROM {self.table_name}
                GROUP BY day, user_id
                ORDER BY day DESC;
            """
            )

            # Add refresh policies (refresh every hour)
            try:
                await conn.execute(
                    """
                    SELECT add_continuous_aggregate_policy('memory_hourly_stats',
                        start_offset => INTERVAL '2 hours',
                        end_offset => INTERVAL '1 hour',
                        schedule_interval => INTERVAL '1 hour',
                        if_not_exists => TRUE);
                """
                )

                await conn.execute(
                    """
                    SELECT add_continuous_aggregate_policy('memory_daily_stats',
                        start_offset => INTERVAL '2 days',
                        end_offset => INTERVAL '1 day',
                        schedule_interval => INTERVAL '1 day',
                        if_not_exists => TRUE);
                """
                )
            except Exception:
                # Policies might already exist
                pass

    def _extract_temporal_data(self, entry: MemoryEntry) -> Dict[str, Any]:
        """Extract temporal information from memory entry"""
        temporal_data = {
            "event_date": None,
            "relative_time": None,
            "time_of_day": None,
            "temporal_context": {},
        }

        # Extract from custom metadata if temporal info was processed
        if "event_date" in entry.custom_metadata:
            temporal_data["event_date"] = entry.custom_metadata["event_date"]

        if "relative_time" in entry.custom_metadata:
            temporal_data["relative_time"] = entry.custom_metadata["relative_time"]

        if "time_of_day" in entry.custom_metadata:
            temporal_data["time_of_day"] = str(entry.custom_metadata["time_of_day"])

        # Store additional temporal context
        temporal_context = {}
        for key in ["days_offset", "temporal_references", "time_extraction_confidence"]:
            if key in entry.custom_metadata:
                temporal_context[key] = entry.custom_metadata[key]

        temporal_data["temporal_context"] = temporal_context

        return temporal_data

    def _extract_graph_summary(self, entry: MemoryEntry) -> Dict[str, Any]:
        """Extract summary of graph relationships"""
        entity_count = len(
            [link for link in entry.graph_links if link.target_id.startswith("entity:")]
        )
        concept_count = len(
            [
                link
                for link in entry.graph_links
                if link.target_id.startswith("concept:")
            ]
        )
        relationship_types = list(
            set([link.relationship_type for link in entry.graph_links])
        )

        return {
            "entity_count": entity_count,
            "concept_count": concept_count,
            "relationship_types": relationship_types,
        }

    async def close(self):
        """Clean shutdown of TimescaleDB store"""
        if self.pool:
            await self.pool.close()
            print("✅ TimescaleDB store closed")

    async def create(self, entry: MemoryEntry) -> str:
        """Create a new memory entry in the time series"""
        temporal_data = self._extract_temporal_data(entry)
        graph_summary = self._extract_graph_summary(entry)

        async with self.pool.acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO {self.table_name} (
                    time, memory_id, user_id, session_id, memory_type, scope,
                    input_text, summary, tags, confidence_score, processing_time_ms,
                    metadata, event_date, relative_time, time_of_day, temporal_context,
                    entity_count, concept_count, relationship_types
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19
                )
            """,
                entry.created_at,
                entry.id,
                entry.source_user_id,
                entry.source_session_id,
                entry.memory_type.value,
                entry.scope,
                entry.input,
                entry.summary,
                entry.tags,
                entry.meta.confidence_score,
                entry.meta.processing_time_ms,
                json.dumps(entry.custom_metadata),
                temporal_data["event_date"],
                temporal_data["relative_time"],
                temporal_data["time_of_day"],
                json.dumps(temporal_data["temporal_context"]),
                graph_summary["entity_count"],
                graph_summary["concept_count"],
                graph_summary["relationship_types"],
            )

        return entry.id

    async def add_memory(self, memory_dict: Dict[str, Any]) -> bool:
        """Add memory from dictionary format with enhanced cross-database references"""
        try:
            # Extract basic temporal data from the dictionary
            created_at = memory_dict.get("created_at", datetime.utcnow())
            if isinstance(created_at, str):
                from dateutil.parser import parse

                created_at = parse(created_at)

            # Enhanced temporal data extraction
            temporal_data = await self._extract_enhanced_temporal_data(
                memory_dict, created_at
            )

            # Enhanced graph summary with cross-references
            graph_summary = await self._extract_enhanced_graph_summary(memory_dict)

            # Extract cross-database references
            cross_references = await self._extract_cross_database_references(
                memory_dict
            )

            async with self.pool.acquire() as conn:
                await conn.execute(
                    f"""
                    INSERT INTO {self.table_name} (
                        time, memory_id, user_id, session_id, memory_type, scope,
                        input_text, summary, tags, confidence_score, processing_time_ms,
                        metadata, 
                        
                        -- Cross-database references
                        vector_store_ref, graph_store_refs, doc_store_ref, audit_store_refs,
                        
                        -- Temporal fields
                        event_date, relative_time, time_of_day, temporal_context, temporal_references,
                        
                        -- Enhanced relationship tracking  
                        entity_count, concept_count, relationship_types, connected_memories, memory_clusters,
                        
                        -- Contextual information
                        temporal_significance, cross_reference_count
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, 
                        $13, $14, $15, $16, 
                        $17, $18, $19, $20, $21, 
                        $22, $23, $24, $25, $26, 
                        $27, $28
                    )
                """,
                    created_at,
                    memory_dict.get("id"),
                    memory_dict.get("source_user_id")
                    or memory_dict.get("meta", {}).get("user_id"),
                    memory_dict.get("source_session_id")
                    or memory_dict.get("meta", {}).get("session_id"),
                    memory_dict.get("memory_type").value
                    if hasattr(memory_dict.get("memory_type"), "value")
                    else str(memory_dict.get("memory_type")),
                    memory_dict.get("scope"),
                    memory_dict.get("input"),
                    memory_dict.get("summary"),
                    memory_dict.get("tags", []),
                    memory_dict.get("meta", {}).get("confidence_score", 0.0),
                    memory_dict.get("meta", {}).get("processing_time_ms", 0),
                    json.dumps(memory_dict.get("custom_metadata", {})),
                    # Cross-database references
                    cross_references.get("vector_store_ref"),
                    cross_references.get("graph_store_refs", []),
                    cross_references.get("doc_store_ref"),
                    cross_references.get("audit_store_refs", []),
                    # Temporal data
                    temporal_data["event_date"],
                    temporal_data["relative_time"],
                    temporal_data["time_of_day"],
                    json.dumps(temporal_data["temporal_context"]),
                    json.dumps(temporal_data.get("temporal_references", {})),
                    # Enhanced graph data
                    graph_summary["entity_count"],
                    graph_summary["concept_count"],
                    graph_summary["relationship_types"],
                    graph_summary.get("connected_memories", []),
                    graph_summary.get("memory_clusters", []),
                    # Contextual information
                    temporal_data.get("temporal_significance", 0.0),
                    len(cross_references.get("graph_store_refs", []))
                    + len(cross_references.get("audit_store_refs", [])),
                )

            return True

        except Exception as e:
            print(f"Error adding memory to TimescaleDB: {e}")
            return False

    async def _extract_enhanced_temporal_data(
        self, memory_dict: Dict[str, Any], created_at: datetime
    ) -> Dict[str, Any]:
        """Extract enhanced temporal data with cross-references"""
        temporal_data = {
            "event_date": created_at.date(),
            "relative_time": "present",
            "time_of_day": str(created_at.hour),
            "temporal_context": {"parsed_from_dict": True},
            "temporal_references": {},
            "temporal_significance": 0.5,  # Default significance
        }

        # Extract temporal significance based on content
        input_text = memory_dict.get("input", "").lower()
        if any(
            word in input_text
            for word in ["important", "milestone", "achievement", "crisis", "emergency"]
        ):
            temporal_data["temporal_significance"] = 0.9
        elif any(
            word in input_text
            for word in ["meeting", "appointment", "deadline", "birthday"]
        ):
            temporal_data["temporal_significance"] = 0.7

        return temporal_data

    async def _extract_enhanced_graph_summary(
        self, memory_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract enhanced graph summary with cross-references"""
        graph_links = memory_dict.get("graph_links", [])

        # Extract connected memory IDs from graph links
        connected_memories = []
        for link in graph_links:
            if isinstance(link, dict) and link.get("target_id", "").startswith(
                "memory:"
            ):
                connected_memories.append(link["target_id"].replace("memory:", ""))

        return {
            "entity_count": len(
                [
                    l
                    for l in graph_links
                    if isinstance(l, dict) and l.get("relationship_type") == "mentions"
                ]
            ),
            "concept_count": len(
                [
                    l
                    for l in graph_links
                    if isinstance(l, dict)
                    and l.get("relationship_type") == "relates_to"
                ]
            ),
            "relationship_types": list(
                set(
                    [
                        l.get("relationship_type")
                        for l in graph_links
                        if isinstance(l, dict) and l.get("relationship_type")
                    ]
                )
            ),
            "connected_memories": connected_memories,
            "memory_clusters": [],  # Would be populated by clustering algorithm
        }

    async def _extract_cross_database_references(
        self, memory_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract cross-database references"""
        memory_id = memory_dict.get("id")

        return {
            "vector_store_ref": f"vec_{memory_id}",  # Reference to vector embedding
            "doc_store_ref": memory_id,  # Same ID in document store
            "graph_store_refs": [f"mem_{memory_id}"],  # Graph node ID
            "audit_store_refs": [],  # Will be populated when audit entries are created
        }

    async def health_check(self) -> bool:
        """Check if TimescaleDB store is healthy"""
        try:
            if not self.pool:
                return False

            # Test connection by running a simple query
            async with self.pool.acquire() as conn:
                result = await conn.fetchval("SELECT 1")
                return result == 1

        except Exception as e:
            print(f"TimescaleDB health check failed: {e}")
            return False

    async def read(self, memory_id: str) -> Optional[MemoryEntry]:
        """Read a memory entry by ID - returns simplified time series data"""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                SELECT * FROM {self.table_name} WHERE memory_id = $1 ORDER BY time DESC LIMIT 1
            """,
                memory_id,
            )

            if row:
                return dict(row)
            return None

    async def update(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """Update a memory entry - limited fields can be updated in time series"""
        updateable_fields = ["summary", "tags", "metadata"]

        set_clauses = []
        params = [memory_id]
        param_idx = 2

        for field, value in updates.items():
            if field in updateable_fields:
                if field == "metadata":
                    set_clauses.append(f"metadata = ${param_idx}")
                    params.append(json.dumps(value))
                else:
                    set_clauses.append(f"{field} = ${param_idx}")
                    params.append(value)
                param_idx += 1

        if not set_clauses:
            return True

        async with self.pool.acquire() as conn:
            result = await conn.execute(
                f"""
                UPDATE {self.table_name} 
                SET {', '.join(set_clauses)}
                WHERE memory_id = $1
            """,
                *params,
            )

            return result != "UPDATE 0"

    async def delete(self, memory_id: str) -> bool:
        """Delete a memory entry from time series"""
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                f"""
                DELETE FROM {self.table_name} WHERE memory_id = $1
            """,
                memory_id,
            )

            return result != "DELETE 0"

    async def search(self, query: Dict[str, Any], limit: int = 10) -> List[MemoryEntry]:
        """Search in time series data"""
        where_conditions = []
        params = []
        param_idx = 1

        for key, value in query.items():
            if key in ["user_id", "memory_type", "session_id"]:
                where_conditions.append(f"{key} = ${param_idx}")
                params.append(value)
                param_idx += 1

        where_clause = " AND ".join(where_conditions) if where_conditions else "TRUE"

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT * FROM {self.table_name}
                WHERE {where_clause}
                ORDER BY time DESC
                LIMIT {limit}
            """,
                *params,
            )

            return [dict(row) for row in rows]

    async def query_time_range(
        self,
        start_time: str,
        end_time: str,
        user_id: Optional[str] = None,
        memory_type: Optional[str] = None,
    ) -> List[MemoryEntry]:
        """Query memories within time range"""
        where_conditions = ["time >= $1", "time <= $2"]
        params = [start_time, end_time]
        param_idx = 3

        if user_id:
            where_conditions.append(f"user_id = ${param_idx}")
            params.append(user_id)
            param_idx += 1

        if memory_type:
            where_conditions.append(f"memory_type = ${param_idx}")
            params.append(memory_type)
            param_idx += 1

        where_clause = " AND ".join(where_conditions)

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT * FROM {self.table_name}
                WHERE {where_clause}
                ORDER BY time ASC
            """,
                *params,
            )

            return [dict(row) for row in rows]

    async def get_timeline(
        self, user_id: str, granularity: str = "day"  # hour, day, week, month
    ) -> List[Dict[str, Any]]:
        """Get aggregated timeline of memories"""

        # Map granularity to TimescaleDB time_bucket intervals
        interval_map = {
            "hour": "1 hour",
            "day": "1 day",
            "week": "1 week",
            "month": "1 month",
        }

        interval = interval_map.get(granularity, "1 day")

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT 
                    time_bucket($1, time) as time_bucket,
                    COUNT(*) as memory_count,
                    array_agg(DISTINCT memory_type) as memory_types,
                    AVG(confidence_score) as avg_confidence,
                    SUM(entity_count) as total_entities,
                    SUM(concept_count) as total_concepts,
                    array_agg(DISTINCT unnest(relationship_types)) as all_relationships
                FROM {self.table_name}
                WHERE user_id = $2
                GROUP BY time_bucket
                ORDER BY time_bucket DESC
                LIMIT 100
            """,
                interval,
                user_id,
            )

            return [dict(row) for row in rows]

    async def get_memory_patterns(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """Analyze memory patterns over time"""
        async with self.pool.acquire() as conn:
            # Daily patterns
            daily_pattern = await conn.fetch(
                """
                SELECT 
                    EXTRACT(hour FROM time) as hour_of_day,
                    COUNT(*) as memory_count,
                    array_agg(DISTINCT memory_type) as common_types
                FROM memory_timeseries
                WHERE user_id = $1 AND time >= NOW() - INTERVAL '%s days'
                GROUP BY hour_of_day
                ORDER BY hour_of_day
            """
                % days,
                user_id,
            )

            # Weekly patterns
            weekly_pattern = await conn.fetch(
                """
                SELECT 
                    EXTRACT(dow FROM time) as day_of_week,
                    COUNT(*) as memory_count,
                    AVG(confidence_score) as avg_confidence
                FROM memory_timeseries
                WHERE user_id = $1 AND time >= NOW() - INTERVAL '%s days'
                GROUP BY day_of_week
                ORDER BY day_of_week
            """
                % days,
                user_id,
            )

            # Memory type trends
            type_trends = await conn.fetch(
                """
                SELECT 
                    time_bucket('1 day', time) as day,
                    memory_type,
                    COUNT(*) as count
                FROM memory_timeseries
                WHERE user_id = $1 AND time >= NOW() - INTERVAL '%s days'
                GROUP BY day, memory_type
                ORDER BY day DESC, count DESC
            """
                % days,
                user_id,
            )

            return {
                "daily_pattern": [dict(row) for row in daily_pattern],
                "weekly_pattern": [dict(row) for row in weekly_pattern],
                "type_trends": [dict(row) for row in type_trends],
                "analysis_period_days": days,
            }

    async def retention_policy(self, policy: Dict[str, Any]):
        """Set data retention policy"""
        async with self.pool.acquire() as conn:
            # Example: Drop chunks older than specified period
            if "drop_older_than" in policy:
                await conn.execute(
                    """
                    SELECT drop_chunks(
                        relation => $1,
                        older_than => $2
                    );
                """,
                    self.table_name,
                    policy["drop_older_than"],
                )

            # Set compression policy
            if "compress_after" in policy:
                try:
                    await conn.execute(
                        f"""
                        SELECT add_compression_policy('{self.table_name}', 
                                                    compress_after => INTERVAL '{policy["compress_after"]}');
                    """
                    )
                except Exception:
                    # Policy might already exist
                    pass

    async def close(self):
        """Close TimescaleDB connection pool"""
        if self.pool:
            await self.pool.close()
