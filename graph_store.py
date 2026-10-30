import json
import os
import re

from dotenv import load_dotenv
from google import genai
from neo4j import GraphDatabase

load_dotenv()

# Gemini API
api_key = os.getenv("GEMINI_API_KEY")
model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

gemini_client = genai.Client(api_key=api_key) if api_key else None

# Neo4j connection
URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "")

_driver = None


def get_driver():
    """Create and verify the Neo4j connection."""
    global _driver

    if not PASSWORD:
        raise ValueError(
            "NEO4J_PASSWORD is missing in your .env file."
        )

    if _driver is None:
        _driver = GraphDatabase.driver(
            URI,
            auth=(USERNAME, PASSWORD),
        )
        _driver.verify_connectivity()

    return _driver


def extract_facts(text):
    """Ask Gemini to extract entities and relationships."""

    if not gemini_client:
        raise ValueError(
            "GEMINI_API_KEY is missing in .env."
        )

    prompt = f"""
Extract important entities and relationships from this text.

Return ONLY valid JSON in this exact structure:
{{
  "entities": [
    {{"name": "Python", "type": "Technology"}}
  ],
  "relationships": [
    {{
      "source": "Python",
      "relation": "USED_FOR",
      "target": "Programming"
    }}
  ]
}}

Rules:
- Use only facts supported by the text.
- Keep entity names short and meaningful.
- Do not invent entities or relationships.
- Use uppercase letters and underscores for relation names.
- Include only entities that appear in the text.

TEXT:
{text[:7000]}
"""

    response = gemini_client.models.generate_content(
        model=model_name,
        contents=prompt,
    )

    raw = (response.text or "").strip()

    # Handle JSON wrapped in Markdown fences.
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    data = json.loads(raw)

    if not isinstance(data.get("entities"), list):
        raise ValueError("Gemini returned invalid entities.")

    if not isinstance(data.get("relationships"), list):
        raise ValueError("Gemini returned invalid relationships.")

    return data


def save_facts(facts, source, page):
    """Store entities and relationships in Neo4j."""

    driver = get_driver()

    with driver.session() as session:
        for entity in facts.get("entities", []):
            name = str(entity.get("name", "")).strip()
            entity_type = str(
                entity.get("type", "Concept")
            ).strip()[:100]

            if not name:
                continue

            session.run(
                """
                MERGE (e:Entity {name: $name})
                SET e.type = $entity_type
                """,
                name=name[:200],
                entity_type=entity_type,
            )

        for rel in facts.get("relationships", []):
            source_name = str(
                rel.get("source", "")
            ).strip()[:200]

            target_name = str(
                rel.get("target", "")
            ).strip()[:200]

            relation = str(
                rel.get("relation", "RELATED_TO")
            ).strip().upper()

            # Keep relationship labels as data, not dynamic Cypher.
            if not re.fullmatch(r"[A-Z][A-Z0-9_]{0,49}", relation):
                relation = "RELATED_TO"

            if not source_name or not target_name:
                continue

            session.run(
                """
                MERGE (a:Entity {name: $source_name})
                MERGE (b:Entity {name: $target_name})
                MERGE (a)-[r:RELATED_TO {kind: $relation}]->(b)
                SET r.source = $source,
                    r.page = $page
                """,
                source_name=source_name,
                target_name=target_name,
                relation=relation,
                source=source,
                page=int(page),
            )


def index_text_in_graph(text, source, page):
    """Extract facts and store them for one PDF page/chunk."""

    if not text or not text.strip():
        return 0

    facts = extract_facts(text)
    save_facts(facts, source, page)

    return len(facts.get("entities", []))


def search_graph(question, limit=10):
    """Find graph facts related to entity names in the question."""

    driver = get_driver()

    with driver.session() as session:
        result = session.run(
            """
            MATCH (a:Entity)-[r:RELATED_TO]->(b:Entity)
            WHERE toLower($question) CONTAINS toLower(a.name)
               OR toLower($question) CONTAINS toLower(b.name)
               OR toLower(a.name) CONTAINS toLower($question)
            RETURN a.name AS source_entity,
                   r.kind AS relationship,
                   b.name AS target_entity,
                   r.source AS source,
                   r.page AS page
            LIMIT $limit
            """,
            question=question,
            limit=limit,
        )

        return [record.data() for record in result]


def close_driver():
    """Close Neo4j connection when needed."""
    global _driver

    if _driver is not None:
        _driver.close()
        _driver = None
        
def get_graph_data():
    """Return nodes and relationships for graph visualization."""
    return [], []


    with driver.session() as session:
        result = session.run(
            """
            MATCH (a:Entity)-[r:RELATED_TO]->(b:Entity)
            RETURN a.name AS source_entity,
                   a.type AS source_type,
                   r.kind AS relationship,
                   b.name AS target_entity,
                   b.type AS target_type,
                   r.source AS source,
                   r.page AS page
            LIMIT $limit
            """,
            limit=limit,
        )

        return [record.data() for record in result]