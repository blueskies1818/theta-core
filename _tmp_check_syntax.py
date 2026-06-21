import py_compile
py_compile.compile('scripts/eval/infer_goal_only.py', doraise=True)
py_compile.compile('scripts/gates/audit_goal_only.py', doraise=True)
py_compile.compile('scripts/training/train_goal_only.py', doraise=True)
py_compile.compile('src/explorer/goal_only_encoder.py', doraise=True)
print('All scripts compile OK')
