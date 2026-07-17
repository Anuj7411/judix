#!/usr/bin/env bash
# Judix stress test — verifies the 3 demos hit their expected bands every time,
# under cold cache, warm cache, and concurrent load, with zero dropped metrics.
#
# Usage:  judix-server running on :8000, then  bash scripts/stress.sh
#
# This exists because single happy-path runs hid two real bugs: under load the
# fast provider throttles, and metrics were silently degrading to `na` while the
# band still looked right. Asserting `na == 0` (not just the band) is the point.
cd "$(dirname "$0")/.." || exit 1

check() { # $1=endpoint $2=file $3=expected_band $4=label
  local out band q na
  out=$(curl -s "http://localhost:8000/$1" -H "Content-Type: application/json" -d @"$2" --max-time 240)
  read -r band q na <<<"$(echo "$out" | python3 -c "
import sys,json
r=json.load(sys.stdin)
q=r.get('run_quality', r.get('rag_quality', -1))
na=sum(1 for s in r.get('steps',[]) for m in s['metrics'] if m.get('na'))+sum(1 for m in r.get('metrics',[]) if m.get('na'))
print(r.get('band','ERR'), round(q,1), na)
" 2>/dev/null || echo "PARSE_ERR -1 -1")"
  if [ "$band" = "$3" ] && [ "$na" = "0" ]; then
    echo "  PASS $4: $q $band (na=$na)"; return 0
  else
    echo "  FAIL $4: got '$band' $q na=$na — expected $3 with na=0"; return 1
  fi
}

FAILS=0
echo "=== ROUND 1: COLD CACHE (sequential) ==="
check score/agent demos/wrong_tool.json red   "wrong_tool" || FAILS=$((FAILS+1))
check score/agent demos/clean.json      green "clean"      || FAILS=$((FAILS+1))
check score/rag   demos/rag_hallucination.json red "rag"   || FAILS=$((FAILS+1))

echo ""
echo "=== ROUND 2: WARM CACHE (must match round 1 exactly) ==="
check score/agent demos/wrong_tool.json red   "wrong_tool" || FAILS=$((FAILS+1))
check score/agent demos/clean.json      green "clean"      || FAILS=$((FAILS+1))
check score/rag   demos/rag_hallucination.json red "rag"   || FAILS=$((FAILS+1))

echo ""
echo "=== ROUND 3: CONCURRENT LOAD (6 simultaneous — simulates multiple judges) ==="
TMP=$(mktemp -d)
for i in 1 2; do
  ( check score/agent demos/wrong_tool.json red   "wrong_tool#$i" > "$TMP/wt$i" 2>&1 ) &
  ( check score/agent demos/clean.json      green "clean#$i"      > "$TMP/cl$i" 2>&1 ) &
  ( check score/rag   demos/rag_hallucination.json red "rag#$i"   > "$TMP/rg$i" 2>&1 ) &
done
wait
for f in "$TMP"/*; do cat "$f"; grep -q "FAIL" "$f" && FAILS=$((FAILS+1)); done
rm -rf "$TMP"

echo ""
echo "=== ROUND 4: SSE (deterministic must arrive BEFORE done, not with it) ==="
sse_check() { # $1=endpoint $2=file $3=first_expected_event $4=label
  local out
  out=$(curl -N -s "http://localhost:8000/$1" -H "Content-Type: application/json" -d @"$2" --max-time 240 \
        | grep '^event:' | sed 's/event: *//' | tr '\n' ' ')
  local first="${out%% *}"
  if [ "$first" = "$3" ] && echo "$out" | grep -q "done"; then
    echo "  PASS $4: $out"; return 0
  else
    echo "  FAIL $4: expected first='$3' then done — got: $out"; return 1
  fi
}
sse_check score/agent/stream demos/wrong_tool.json deterministic "agent SSE" || FAILS=$((FAILS+1))
sse_check score/rag/stream   demos/rag_hallucination.json claims     "rag SSE"   || FAILS=$((FAILS+1))

echo ""
echo "==================================="
if [ "$FAILS" -eq 0 ]; then echo "ALL PASS — 14/14 checks (12 scoring + 2 SSE), zero dropped metrics"; else echo "$FAILS CHECK(S) FAILED"; fi
echo "==================================="
