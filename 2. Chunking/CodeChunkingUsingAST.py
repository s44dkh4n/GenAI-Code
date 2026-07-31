import ast
from typing import List, Dict, Any
from pydantic import BaseModel, Field

# Target source code string to ingest into RAG
sample_code = """
import os

class RateLimiter:
    \"\"\"Manages API request limits per user token.\"\"\"
    def __init__(self, max_requests: int = 100):
        self.max_requests = max_requests

    def check_limit(self, user_id: str) -> bool:
        # Check current count against ceiling
        return True

def process_payload(data: dict) -> dict:
    \"\"\"Processes and cleans raw JSON payload.\"\"\"
    return {"status": "success", "data": data}
"""

# Pydantic schema for structured code chunks
class CodeChunk(BaseModel):
    name: str = Field(description="Name of the function or class")
    node_type: str = Field(description="AST Node classification (ClassDef or FunctionDef)")
    code_body: str = Field(description="Untruncated source code block")
    docstring: str = Field(default="", description="Extracted docstring for vector indexing")
    line_start: int = Field(description="Starting line in source file")
    line_end: int = Field(description="Ending line in source file")

def parse_code_with_ast(source_code: str) -> List[CodeChunk]:
    # Parse source string into an Abstract Syntax Tree
    tree = ast.parse(source_code)
    source_lines = source_code.splitlines()
    chunks: List[CodeChunk] = []

    # Iterate top-level nodes in the module
    for node in tree.body:
        # Handle Class Definitions
        if isinstance(node, ast.ClassDef):
            class_code = "\n".join(source_lines[(node.lineno) - 1 : node.end_lineno])
            chunks.append(
                CodeChunk(
                    name=node.name,
                    node_type="ClassDef",
                    code_body=class_code,
                    docstring=ast.get_docstring(node) or "",
                    line_start=node.lineno,
                    line_end=node.end_lineno
                )
            )
        # Handle Top-Level Function Definitions
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_code = "\n".join(source_lines[(node.lineno) - 1 : node.end_lineno])
            chunks.append(
                CodeChunk(
                    name=node.name,
                    node_type="FunctionDef",
                    code_body=func_code,
                    docstring=ast.get_docstring(node) or "",
                    line_start=node.lineno,
                    line_end=node.end_lineno
                )
            )

    return chunks

# Execute parser
parsed_chunks = parse_code_with_ast(sample_code)

# Display parsed structural code units
for chunk in parsed_chunks:
    print(f"Name: {chunk.name} ({chunk.node_type})")
    print(f"Docstring: {chunk.docstring}")
    print(f"Lines: {chunk.line_start} to {chunk.line_end}")
    print(f"Body:\n{chunk.code_body}")
    print("-" * 40)