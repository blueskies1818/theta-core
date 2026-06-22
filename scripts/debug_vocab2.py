import sys
sys.path.insert(0, '/home/blueman1818/Projects/theta-core')
from src.physics.composer import TEMPLATE_ID_TO_TOKEN, TEMPLATE_TOKEN_TO_ID

print(f'TEMPLATE_TOKEN_TO_ID count: {len(TEMPLATE_TOKEN_TO_ID)}')
print()
for i in range(len(TEMPLATE_TOKEN_TO_ID)):
    tok = TEMPLATE_ID_TO_TOKEN.get(i, "MISSING")
    print(f'  {i}: "{tok}"')
print()
# Show tokens near the edge
print("Tokens 50-60:")
for i in range(50, 60):
    print(f'  {i}: "{TEMPLATE_ID_TO_TOKEN.get(i, "MISSING")}"')
