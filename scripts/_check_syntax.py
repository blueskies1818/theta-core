#!/usr/bin/env python3
import ast, sys
for f in ["scripts/verify_8_claims.py", "src/physics/auto_lean.py"]:
    try:
        ast.parse(open(f).read())
        print(f"{f}: Syntax OK")
    except SyntaxError as e:
        print(f"{f}: Syntax error: {e}")
        sys.exit(1)
