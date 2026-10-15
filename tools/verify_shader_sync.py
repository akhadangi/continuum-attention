"""Verify the browsable .metal copy matches the runtime-embedded shader."""
from pathlib import Path
import ast

base = Path(__file__).resolve().parents[1]
tree = ast.parse((base / 'acma_metal.py').read_text())
embedded = next(
    node.value.value for node in tree.body
    if isinstance(node, ast.Assign)
    and any(isinstance(target, ast.Name) and target.id == 'METAL_SOURCE' for target in node.targets)
    and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
)
external = (base / 'metal' / 'continuum_attention.metal').read_text()
if embedded != external:
    raise SystemExit('Embedded and standalone Metal source differ')
print('PASS: browsable .metal file exactly matches runtime shader')
