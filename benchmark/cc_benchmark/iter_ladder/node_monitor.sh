#!/bin/bash
# Lightweight shared-node contention sampler. Runs in the background alongside a fit and writes a
# timestamped CSV of node-wide load + CO-TENANT footprint (other users' reserved cores on our node)
# so the time-vs-iteration variance can be attributed to contention. perf_event_paranoid=2 on fir
# blocks system-wide/uncore bandwidth counters, so co-tenant reserved cores + node CPU-busy% +
# loadavg are our best unprivileged bandwidth proxies.
#
#   node_monitor.sh <out.csv> [interval_s] [node]
# Stops when signalled (the caller kills it after the fit).
set -u
OUT="${1:?out csv path}"; INT="${2:-10}"; NODE="${3:-${SLURMD_NODENAME:-$(hostname -s)}}"
ME="${USER}"
echo "epoch,node,loadavg1,node_cpu_busy_pct,mem_used_gb,ncpu_total,cotenant_jobs,cotenant_cores,own_cores" > "$OUT"

# prev total/idle jiffies for busy% delta
prev=$(awk '/^cpu /{t=0;for(i=2;i<=NF;i++)t+=$i; print t, $5+$6}' /proc/stat)   # total, idle+iowait
trap 'exit 0' TERM INT
while :; do
  now=$(awk '/^cpu /{t=0;for(i=2;i<=NF;i++)t+=$i; print t, $5+$6}' /proc/stat)
  busy=$(awk -v p="$prev" -v n="$now" 'BEGIN{split(p,P);split(n,N);dt=N[1]-P[1];di=N[2]-P[2];
         if(dt>0) printf "%.1f",100*(dt-di)/dt; else printf "0"}')
  prev="$now"
  la=$(awk '{print $1}' /proc/loadavg)
  ncpu=$(nproc --all 2>/dev/null || echo 0)
  memused=$(awk '/MemTotal/{t=$2}/MemAvailable/{a=$2}END{printf "%.1f",(t-a)/1048576}' /proc/meminfo)
  # co-tenants = jobs on this node belonging to OTHER users (%C = allocated CPUs)
  read cj cc oc < <(squeue -w "$NODE" -h -o "%u %C" 2>/dev/null | awk -v me="$ME" '
       {if($1!=me){cj++; cc+=$2} else {oc+=$2}} END{printf "%d %d %d", cj+0, cc+0, oc+0}')
  printf "%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
     "$(date +%s)" "$NODE" "$la" "$busy" "$memused" "$ncpu" "${cj:-0}" "${cc:-0}" "${oc:-0}" >> "$OUT"
  sleep "$INT"
done
