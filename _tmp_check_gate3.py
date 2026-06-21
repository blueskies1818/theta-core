import json
d = json.load(open('data/goal_only_gate3.json'))
print(f"Passed: {d.get('passed','?')}/{d.get('total','?')} = {d.get('rate','?')}")
