#!/bin/bash
# Check Python process stack
PID=297031
# Try py-spy
if command -v py-spy &>/dev/null; then
    timeout 10 py-spy dump --pid $PID 2>&1
    exit 0
fi
# Try gdb
if command -v gdb &>/dev/null; then
    timeout 10 gdb -p $PID -batch -ex "thread apply all py-bt" 2>&1 | head -50
    exit 0
fi
echo "No debugging tools available"
