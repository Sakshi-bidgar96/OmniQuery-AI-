from dotenv import load_dotenv
from neo4j import GraphDatabase
import os

load_dotenv()

uri = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")

try:
    driver = GraphDatabase.driver(
        uri,
        auth=(username, password),
    )

    driver.verify_connectivity()
    print("SUCCESS: Neo4j connection working!")

    with driver.session(database="neo4j") as session:
        result = session.run(
            "RETURN 'Graph RAG Connected' AS message"
        )
        print(result.single()["message"])

    driver.close()

except Exception as error:
    print("Connection failed:", error)