#!/bin/sh
# Chain: wait for the standard sweep, then run deep. Deep ADDS to results --
# steps is part of the run identity, so nothing already finished is repeated.
cd "$(dirname "$0")"
PY=./.venv/Scripts/python.exe

echo "[$(date +%H:%M:%S)] waiting for the standard sweep to finish ..."
while [ "$(powershell.exe -NoProfile -Command '(Get-Process python -EA SilentlyContinue|Measure-Object).Count' 2>/dev/null | tr -d '\r')" != "0" ]; do
    sleep 60
done

echo "[$(date +%H:%M:%S)] standard sweep done; $(cat outputs/sweep/results_shard*.jsonl 2>/dev/null | wc -l) runs recorded"
$PY sweep.py --summary-only > outputs/sweep/summary_standard.txt 2>&1
echo "[$(date +%H:%M:%S)] standard summary saved to outputs/sweep/summary_standard.txt"

echo "[$(date +%H:%M:%S)] starting DEEP sweep (~19.5 h)"
$PY -u sweep.py --deep --workers 10 >> outputs/sweep/overnight.log 2>&1
echo "[$(date +%H:%M:%S)] deep sweep finished"
$PY sweep.py --summary-only > outputs/sweep/summary_deep.txt 2>&1
echo "[$(date +%H:%M:%S)] all done"
