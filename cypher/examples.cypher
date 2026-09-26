// Example Cypher queries for exploring the GraphRAG knowledge graph.
// Open Neo4j Browser at http://localhost:7474  (user: neo4j, password from .env)
// and paste any of these.

// 1. Overall size of the graph
MATCH (n)
RETURN labels(n) AS label, count(*) AS count
ORDER BY count DESC;

// 2. The most connected entities (hubs)
MATCH (e:Entity)-[r:RELATES_TO]-()
RETURN e.display_name AS entity, e.type AS type, count(r) AS degree
ORDER BY degree DESC
LIMIT 20;

// 3. Top relationships by how often they appeared across chunks
MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
RETURN a.display_name AS source, r.description AS relationship,
       b.display_name AS target, r.weight AS weight
ORDER BY weight DESC
LIMIT 25;

// 4. Everything connected to one entity (local neighbourhood)
MATCH (e:Entity {name: "manchester united"})-[r:RELATES_TO]-(other:Entity)
RETURN e.display_name, r.description, other.display_name;

// 5. Which chunks mention a given entity (evidence / provenance)
MATCH (c:Chunk)-[:MENTIONS]->(e:Entity {name: "ed wood"})
RETURN c.title, c.text;

// 6. A visual subgraph: 2 hops from an entity (switch Browser to "Graph" view)
MATCH p = (e:Entity {name: "scott derrickson"})-[:RELATES_TO*1..2]-(:Entity)
RETURN p
LIMIT 50;

// 7. Community reports (global / thematic summaries)
MATCH (cm:Community)
RETURN cm.title AS community, cm.size AS size, cm.summary AS summary
ORDER BY size DESC
LIMIT 10;

// 8. Community membership
MATCH (e:Entity)-[:IN_COMMUNITY]->(cm:Community)
RETURN cm.title AS community, collect(e.display_name) AS members
ORDER BY size(members) DESC
LIMIT 5;
