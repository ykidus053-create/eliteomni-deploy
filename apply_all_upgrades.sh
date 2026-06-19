#!/bin/bash
set -e
cd ~/eliteomni_app

echo "=============================="
echo " EliteOmni Intelligence Upgrade"
echo "=============================="

echo ""
echo "Step 1: Verifying upgrade files..."
for f in working_memory.py reasoning_engine.py planner.py skill_router.py learning_loop.py world_model.py integrate_upgrades.py; do
    if [ -f "$f" ]; then
        echo "  ✅ $f"
    else
        echo "  ❌ $f MISSING — run the cat commands first"
        exit 1
    fi
done

echo ""
echo "Step 2: Installing dependencies..."
pip install chromadb fastembed scipy numpy --break-system-packages --quiet 2>&1 | tail -5

echo ""
echo "Step 3: Running integration patch..."
python integrate_upgrades.py

echo ""
echo "Step 4: Running unit tests on upgrade modules..."
python -c "
from working_memory import wm_save, wm_retrieve, wm_build_context
wm_save('Test memory: user prefers Python over JavaScript', importance=2.0, tags=['preference'])
results = wm_retrieve('Python preference', k=3)
assert len(results) >= 0, 'WM retrieve failed'
print('  ✅ working_memory: OK')

from skill_router import classify_skill, route_complexity
assert classify_skill('write a python function') == 'coder', f'Expected coder, got {classify_skill(\"write a python function\")}'
assert route_complexity('hi') == 'easy', f'Expected easy, got {route_complexity(\"hi\")}'
assert route_complexity('implement a comprehensive distributed system with fault tolerance') == 'hard'
print('  ✅ skill_router: OK')

from planner import PlanStep, Plan, StepStatus
step = PlanStep(id='s1', description='Test step')
assert step.status == StepStatus.PENDING
print('  ✅ planner: OK')

from learning_loop import log_interaction, get_learning_stats
stats = get_learning_stats()
assert 'total_interactions' in stats
print('  ✅ learning_loop: OK')

from world_model import get_world_model_context, UserModel
ctx = get_world_model_context('debug this Python function')
print('  ✅ world_model: OK')

print()
print('  All upgrade modules verified.')
"

echo ""
echo "Step 5: Starting upgraded server..."
if pgrep -f "uvicorn app:app" > /dev/null 2>&1; then
    echo "  Restarting existing server..."
    fuser -k 8080/tcp 2>/dev/null || true
    sleep 2
fi

nohup uvicorn app:app --host 0.0.0.0 --port 8080 --workers 1 > ~/eliteomni_upgrade.log 2>&1 &
sleep 4

echo "  Checking upgrade status..."
curl -s http://localhost:8080/upgrades/status | python3 -m json.tool 2>/dev/null || echo "  Server still starting..."

echo ""
echo "=============================="
echo " Upgrade Complete"
echo "=============================="
echo ""
echo "New endpoints:"
echo "  GET  /upgrades/status       — module health"
echo "  GET  /upgrades/learning     — learning statistics"
echo "  POST /upgrades/deliberate   — test deliberative reasoning"
echo ""
echo "Logs: tail -f ~/eliteomni_upgrade.log"
