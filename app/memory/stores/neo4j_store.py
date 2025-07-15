import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from neo4j import AsyncGraphDatabase, GraphDatabase

from app.memory.models.memory_entry import GraphLink, MemoryEntry
from app.memory.stores.base_store import GraphStore


class Neo4jGraphStore(GraphStore):
    """Neo4j implementation of GraphStore"""

    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j"):
        """Initialize Neo4j connection"""
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self.driver = None

    async def initialize(self):
        """Initialize Neo4j connection"""
        self.driver = AsyncGraphDatabase.driver(
            self.uri, auth=(self.username, self.password)
        )

        # Create constraints and indexes
        await self._create_schema()

    async def _create_schema(self):
        """Create Neo4j schema - constraints and indexes"""
        async with self.driver.session(database=self.database) as session:
            # Create constraints
            constraints = [
                "CREATE CONSTRAINT memory_id_unique IF NOT EXISTS FOR (m:Memory) REQUIRE m.id IS UNIQUE",
                "CREATE CONSTRAINT user_id_unique IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE",
                "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
                "CREATE CONSTRAINT concept_id_unique IF NOT EXISTS FOR (c:Concept) REQUIRE c.id IS UNIQUE",
                "CREATE CONSTRAINT person_id_unique IF NOT EXISTS FOR (p:Person) REQUIRE p.id IS UNIQUE",
                "CREATE CONSTRAINT location_id_unique IF NOT EXISTS FOR (l:Location) REQUIRE l.id IS UNIQUE",
                "CREATE CONSTRAINT date_id_unique IF NOT EXISTS FOR (d:Date) REQUIRE d.id IS UNIQUE",
                "CREATE CONSTRAINT temporal_id_unique IF NOT EXISTS FOR (t:Temporal) REQUIRE t.id IS UNIQUE",
                "CREATE CONSTRAINT memory_ref_id_unique IF NOT EXISTS FOR (mr:MemoryReference) REQUIRE mr.id IS UNIQUE",
                "CREATE CONSTRAINT meta_state_id_unique IF NOT EXISTS FOR (ms:MetaState) REQUIRE ms.id IS UNIQUE",
                "CREATE CONSTRAINT temporal_meta_id_unique IF NOT EXISTS FOR (tm:TemporalMeta) REQUIRE tm.id IS UNIQUE",
            ]

            for constraint in constraints:
                try:
                    await session.run(constraint)
                except Exception as e:
                    # Constraint might already exist
                    pass

            # Create indexes for performance
            indexes = [
                "CREATE INDEX memory_user_id IF NOT EXISTS FOR (m:Memory) ON m.user_id",
                "CREATE INDEX memory_created_at IF NOT EXISTS FOR (m:Memory) ON m.created_at",
                "CREATE INDEX memory_type IF NOT EXISTS FOR (m:Memory) ON m.memory_type",
            ]

            for index in indexes:
                try:
                    await session.run(index)
                except Exception as e:
                    # Index might already exist
                    pass

    async def create(self, entry: MemoryEntry) -> str:
        """Create a memory node and its relationships"""
        async with self.driver.session(database=self.database) as session:
            # First, create or merge the User node
            user_query = """
            MERGE (u:User {id: $user_id})
            ON CREATE SET u.created_at = datetime()
            SET u.last_active = datetime()
            """
            await session.run(user_query, {"user_id": entry.source_user_id})

            # Create memory node and connect it to the user
            memory_query = """
            MATCH (u:User {id: $user_id})
            CREATE (m:Memory {
                id: $id,
                cid: $cid,
                scope: $scope,
                input: $input,
                summary: $summary,
                memory_type: $memory_type,
                user_id: $user_id,
                session_id: $session_id,
                created_at: $created_at,
                tags: $tags,
                confidence: $confidence
            })
            CREATE (u)-[:HAS_MEMORY]->(m)
            RETURN m.id
            """

            result = await session.run(
                memory_query,
                {
                    "id": entry.id,
                    "cid": entry.cid,
                    "scope": entry.scope,
                    "input": entry.input,
                    "summary": entry.summary,
                    "memory_type": entry.memory_type.value,
                    "user_id": entry.source_user_id,
                    "session_id": entry.source_session_id,
                    "created_at": entry.created_at.isoformat(),
                    "tags": entry.tags,
                    "confidence": entry.meta.confidence_score,
                },
            )

            # Create relationships
            for link in entry.graph_links:
                await self._create_relationship(session, entry.id, link)

            return entry.id

    async def add_memory(self, memory_dict: Dict[str, Any]) -> bool:
        """Add memory from dictionary format (used by memory engine)"""
        try:
            async with self.driver.session(database=self.database) as session:
                # Get user_id
                user_id = memory_dict.get("source_user_id") or memory_dict.get(
                    "meta", {}
                ).get("user_id")

                # First, create or merge the User node
                user_query = """
                MERGE (u:User {id: $user_id})
                ON CREATE SET u.created_at = datetime()
                SET u.last_active = datetime()
                """
                await session.run(user_query, {"user_id": user_id})

                # Create memory node and connect it to the user
                memory_query = """
                MATCH (u:User {id: $user_id})
                CREATE (m:Memory {
                    id: $id,
                    cid: $cid,
                    scope: $scope,
                    input: $input,
                    summary: $summary,
                    memory_type: $memory_type,
                    user_id: $user_id,
                    session_id: $session_id,
                    created_at: $created_at,
                    tags: $tags,
                    confidence: $confidence
                })
                CREATE (u)-[:HAS_MEMORY]->(m)
                RETURN m.id
                """

                # Prepare parameters from dictionary
                params = {
                    "id": memory_dict.get("id"),
                    "cid": memory_dict.get("cid"),
                    "scope": memory_dict.get("scope"),
                    "input": memory_dict.get("input"),
                    "summary": memory_dict.get("summary"),
                    "memory_type": memory_dict.get("memory_type"),
                    "user_id": memory_dict.get("source_user_id")
                    or memory_dict.get("meta", {}).get("user_id"),
                    "session_id": memory_dict.get("source_session_id")
                    or memory_dict.get("meta", {}).get("session_id"),
                    "created_at": memory_dict.get("created_at").isoformat()
                    if hasattr(memory_dict.get("created_at"), "isoformat")
                    else str(memory_dict.get("created_at")),
                    "tags": memory_dict.get("tags", []),
                    "confidence": memory_dict.get("meta", {}).get("confidence_score"),
                }

                # Convert memory_type enum to string if needed
                if hasattr(params.get("memory_type"), "value"):
                    params["memory_type"] = params["memory_type"].value

                result = await session.run(memory_query, params)

                # Create relationships if they exist
                graph_links = memory_dict.get("graph_links", [])
                for link_dict in graph_links:
                    if isinstance(link_dict, dict):
                        # Create a simple GraphLink-like object for compatibility
                        class SimpleLink:
                            def __init__(self, data):
                                self.target_id = data.get("target_id")
                                self.relationship_type = data.get("relationship_type")
                                self.properties = data.get("properties", {})

                        link = SimpleLink(link_dict)
                        await self._create_relationship(
                            session, memory_dict.get("id"), link
                        )

                return True

        except Exception as e:
            print(f"Error adding memory to Neo4j: {e}")
            return False

    async def health_check(self) -> bool:
        """Check if Neo4j store is healthy"""
        try:
            if not self.driver:
                return False

            # Test connection by running a simple query
            async with self.driver.session(database=self.database) as session:
                result = await session.run("RETURN 1 as test")
                record = await result.single()
                return record is not None

        except Exception as e:
            print(f"Neo4j health check failed: {e}")
            return False

    async def _create_relationship(self, session, memory_id: str, link: GraphLink):
        """Create a relationship between memory and target entity"""
        # Determine target node type from target_id
        target_parts = link.target_id.split(":")
        if len(target_parts) != 2:
            return

        target_type, target_name = target_parts

        # Map target types to Neo4j node labels
        label_map = {
            "entity": "Entity",
            "concept": "Concept",
            "person": "Person",
            "location": "Location",
            "date": "Date",
            "temporal": "Temporal",
            "memory_reference": "MemoryReference",
            "meta_state": "MetaState",
            "temporal_meta": "TemporalMeta",
            "tool": "Tool",
            "procedure_type": "ProcedureType",
            "timeline": "Timeline",
        }

        target_label = label_map.get(target_type, "Entity")

        # Create target node if it doesn't exist
        create_target_query = f"""
        MERGE (t:{target_label} {{id: $target_id, name: $target_name}})
        """

        await session.run(
            create_target_query,
            {"target_id": link.target_id, "target_name": target_name},
        )

        # Create relationship
        relationship_query = f"""
        MATCH (m:Memory {{id: $memory_id}})
        MATCH (t:{target_label} {{id: $target_id}})
        CREATE (m)-[r:{link.relationship_type.upper().replace(' ', '_')} $properties]->(t)
        """

        await session.run(
            relationship_query,
            {
                "memory_id": memory_id,
                "target_id": link.target_id,
                "properties": link.properties,
            },
        )

    async def read(self, memory_id: str) -> Optional[MemoryEntry]:
        """Read a memory node - returns simplified data"""
        async with self.driver.session(database=self.database) as session:
            query = """
            MATCH (m:Memory {id: $memory_id})
            RETURN m
            """

            result = await session.run(query, {"memory_id": memory_id})
            record = await result.single()

            if record:
                return record["m"]
            return None

    async def update(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """Update a memory node"""
        async with self.driver.session(database=self.database) as session:
            # Build SET clause dynamically
            set_clauses = []
            params = {"memory_id": memory_id}

            for key, value in updates.items():
                if key in ["tags", "summary", "confidence"]:
                    set_clauses.append(f"m.{key} = ${key}")
                    params[key] = value

            if not set_clauses:
                return True

            query = f"""
            MATCH (m:Memory {{id: $memory_id}})
            SET {', '.join(set_clauses)}
            RETURN m.id
            """

            result = await session.run(query, params)
            record = await result.single()
            return record is not None

    async def delete(self, memory_id: str) -> bool:
        """Delete a memory node and its relationships"""
        async with self.driver.session(database=self.database) as session:
            query = """
            MATCH (m:Memory {id: $memory_id})
            DETACH DELETE m
            RETURN count(m) as deleted_count
            """

            result = await session.run(query, {"memory_id": memory_id})
            record = await result.single()
            return record["deleted_count"] > 0

    async def close(self):
        """Clean shutdown of Neo4j store"""
        if self.driver:
            await self.driver.close()
            print("✅ Neo4j store closed")

    async def search(self, query: Dict[str, Any], limit: int = 10) -> List[MemoryEntry]:
        """Search memories using Cypher query conditions"""
        async with self.driver.session(database=self.database) as session:
            # Build WHERE clause
            where_conditions = []
            params = {"limit": limit}

            # If searching by user_id, use the relationship
            if "user_id" in query:
                cypher_query = """
                MATCH (u:User {id: $user_id})-[:HAS_MEMORY]->(m:Memory)
                """
                params["user_id"] = query["user_id"]
            else:
                cypher_query = "MATCH (m:Memory)"

            if "memory_type" in query:
                where_conditions.append("m.memory_type = $memory_type")
                params["memory_type"] = query["memory_type"]

            if "tags" in query:
                where_conditions.append("ANY(tag IN $tags WHERE tag IN m.tags)")
                params["tags"] = query["tags"]

            where_clause = " AND ".join(where_conditions) if where_conditions else ""

            if where_clause:
                cypher_query += f" WHERE {where_clause}"

            cypher_query += """
            RETURN m
            ORDER BY m.created_at DESC
            LIMIT $limit
            """

            result = await session.run(cypher_query, params)
            records = await result.data()

            return [record["m"] for record in records]

    async def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Add a relationship between two nodes"""
        async with self.driver.session(database=self.database) as session:
            # Determine node types
            source_label = (
                "Memory"
                if not source_id.startswith(("entity:", "concept:"))
                else "Entity"
            )
            target_label = (
                "Memory"
                if not target_id.startswith(("entity:", "concept:"))
                else "Entity"
            )

            query = f"""
            MATCH (s:{source_label} {{id: $source_id}})
            MATCH (t:{target_label} {{id: $target_id}})
            CREATE (s)-[r:{relationship_type.upper().replace(' ', '_')} $properties]->(t)
            RETURN r
            """

            result = await session.run(
                query,
                {
                    "source_id": source_id,
                    "target_id": target_id,
                    "properties": properties or {},
                },
            )

            record = await result.single()
            return record is not None

    async def get_neighbors(
        self, node_id: str, relationship_type: Optional[str] = None, depth: int = 1
    ) -> List[Dict[str, Any]]:
        """Get neighboring nodes"""
        async with self.driver.session(database=self.database) as session:
            # Build relationship filter
            rel_filter = (
                f":{relationship_type.upper().replace(' ', '_')}"
                if relationship_type
                else ""
            )

            query = f"""
            MATCH (start {{id: $node_id}})-[r{rel_filter}*1..{depth}]-(neighbor)
            RETURN DISTINCT neighbor, r, 
                   length((start)-[r{rel_filter}*]-(neighbor)) as distance
            ORDER BY distance
            """

            result = await session.run(query, {"node_id": node_id})
            records = await result.data()

            neighbors = []
            for record in records:
                neighbors.append(
                    {
                        "node": record["neighbor"],
                        "relationships": record["r"],
                        "distance": record["distance"],
                    }
                )

            return neighbors

    async def shortest_path(
        self, source_id: str, target_id: str
    ) -> Optional[List[str]]:
        """Find shortest path between nodes"""
        async with self.driver.session(database=self.database) as session:
            query = """
            MATCH (source {id: $source_id}), (target {id: $target_id})
            MATCH path = shortestPath((source)-[*]-(target))
            RETURN [node in nodes(path) | node.id] as path_ids
            """

            result = await session.run(
                query, {"source_id": source_id, "target_id": target_id}
            )

            record = await result.single()
            if record:
                return record["path_ids"]
            return None

    async def get_memory_context(
        self, memory_id: str, context_depth: int = 2
    ) -> Dict[str, Any]:
        """Get rich context around a memory including related entities, concepts, and other memories"""
        async with self.driver.session(database=self.database) as session:
            query = f"""
            MATCH (m:Memory {{id: $memory_id}})
            OPTIONAL MATCH (m)-[r1*1..{context_depth}]-(related)
            RETURN m as center_memory,
                   collect(DISTINCT related) as related_nodes,
                   collect(DISTINCT r1) as relationships
            """

            result = await session.run(query, {"memory_id": memory_id})
            record = await result.single()

            if record:
                return {
                    "center_memory": record["center_memory"],
                    "related_nodes": record["related_nodes"],
                    "relationships": record["relationships"],
                    "context_depth": context_depth,
                }
            return {}

    async def get_user_knowledge_graph(
        self, user_id: str, limit: int = 100
    ) -> Dict[str, Any]:
        """Get the knowledge graph for a specific user"""
        async with self.driver.session(database=self.database) as session:
            query = """
            MATCH (u:User {id: $user_id})-[:HAS_MEMORY]->(m:Memory)-[r]-(related)
            RETURN u, m, r, related
            LIMIT $limit
            """

            result = await session.run(query, {"user_id": user_id, "limit": limit})
            records = await result.data()

            # Structure the graph data
            nodes = set()
            edges = []
            user_node_added = False

            for record in records:
                user = record["u"]
                memory = record["m"]
                related = record["related"]
                relationship = record["r"]

                # Add user node once
                if not user_node_added:
                    nodes.add((user["id"], "User", f"User: {user['id']}"))
                    user_node_added = True

                # Add memory node
                nodes.add(
                    (
                        memory["id"],
                        "Memory",
                        memory.get("summary", memory.get("input", "")[:100]),
                    )
                )

                # Add related node
                nodes.add(
                    (
                        related["id"],
                        related.labels[0] if related.labels else "Node",
                        related.get("name", related["id"]),
                    )
                )

                # Add user-memory relationship
                edges.append(
                    {
                        "source": user["id"],
                        "target": memory["id"],
                        "relationship": "HAS_MEMORY",
                        "properties": {},
                    }
                )

                # Add memory-related relationship
                edges.append(
                    {
                        "source": memory["id"],
                        "target": related["id"],
                        "relationship": relationship.type,
                        "properties": dict(relationship),
                    }
                )

            return {
                "nodes": [
                    {"id": node[0], "type": node[1], "label": node[2]} for node in nodes
                ],
                "edges": edges,
                "user_id": user_id,
            }

    async def get_total_node_count(self) -> int:
        """Get total number of nodes in the graph"""
        try:
            async with self.driver.session(database=self.database) as session:
                result = await session.run("MATCH (n) RETURN count(n) as total")
                record = await result.single()
                return record["total"] if record else 0
        except Exception as e:
            print(f"Error counting nodes: {e}")
            return 0

    async def close(self):
        """Close Neo4j driver"""
        if self.driver:
            await self.driver.close()
