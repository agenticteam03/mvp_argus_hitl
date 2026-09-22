---
id: RB-002
title: Container restart loop and OOM kills
incident_class: container_restart_loop
severity: critical
host: devops-platform
memory_budget: 3.7 GiB RAM + 2 GiB swap, 2 vCPU
alert_source: cadvisor — container_start_time_seconds churn, container_last_seen
max_rung: 2
agent_may_act_on: containers matching lab-*
related: [RB-001, RB-003]
---

# RB-002 — Container restart loop / OOM kill

## Symptom

A container repeatedly exits and comes back. Every service on this host is
declared `restart: unless-stopped`, so Docker restarts it silently and forever —
**there is no crash notification.** The only outward signs are a climbing
`RestartCount`, a `container_start_time_seconds` that keeps resetting, and
intermittent 502s from the proxy if the flapping container is `backend` or
`frontend`.

A loop can run for days unnoticed. This is the failure mode most worth
automating, precisely because nothing alerts on it today.

## Critical host context — read before diagnosing

**No container on this host has a memory limit.** No compose file sets
`mem_limit` or `deploy.resources.limits`, which is why `docker stats` reports
`3.725GiB` — the host total — as every container's "LIMIT".

The consequence matters for diagnosis: with no per-container cgroup ceiling,
there is no per-container OOM. Instead the **kernel** OOM killer fires host-wide
and picks its victim by `oom_score`, which is driven largely by resident set
size. The process it kills is the *largest*, not necessarily the *leaking* one.

At snapshot the largest RSS processes were `dockerd` (170 MB), the anglemaze Node
backend (86 MB), and `containerd` (68 MB). So a runaway allocation in any
container can get `dockerd` killed, which takes down all six containers at once.

**Therefore: the container that died is not automatically the container at
fault.** Always check `dmesg` for what the kernel actually killed before blaming
the container that restarted.

## Confirm — read-only, rung 1

```bash
docker ps -a --format '{{.Names}}\t{{.Status}}'
docker inspect <name> --format '{{.RestartCount}} {{.State.ExitCode}} {{.State.OOMKilled}}'
docker events --since 1h --filter event=die --filter event=oom
dmesg -T | grep -iE 'killed process|out of memory'
journalctl -k --since '1 hour ago' | grep -i oom
free -h
docker stats --no-stream
docker logs --tail 200 <name>
```

Baseline for comparison: 767 MiB used of 3.7 GiB, **0 B of 2 GiB swap in use**,
load average 0.00. Any swap usage at all is a departure from normal here.

### Reading the exit code

| Exit code | Meaning |
|---|---|
| **137** | SIGKILL (128+9). OOM kill **or** an external `docker kill`. Ambiguous — must be confirmed. |
| **143** | SIGTERM (128+15). Graceful stop; usually a deploy, not a fault. |
| **1** | Application error. Read the logs; this is not a memory problem. |
| **0** | Clean exit of a process that should not have exited — check the entrypoint. |

Disambiguate 137 with `{{.State.OOMKilled}}`. If that is `false` and `dmesg`
shows no kill, something outside Docker sent the signal.

## Likely causes, in order for this host

### 1. Memory exhaustion, host-wide

3.7 GiB across six containers plus `dockerd`, `containerd`, `postgres` and the
system. There is very little headroom, and no per-container cap to contain a
leak.

**Discriminating evidence:** `dmesg` shows "Out of memory: Killed process";
`free -h` shows swap in use; the kill affects a container that was merely large,
not the one whose memory was climbing.

### 2. Application crash on startup

Container starts, throws, exits non-zero, Docker restarts it, repeat. Fast loop —
`RestartCount` climbs by tens per minute.

**Discriminating evidence:** exit code 1, a stack trace in `docker logs`, and
`OOMKilled: false`.

### 3. Failed dependency at boot

`anglemaze-platform-backend-1` declares `depends_on: db: condition:
service_healthy`. The db healthcheck is `pg_isready`, 5 s interval, 5 retries. If
Postgres is slow or refusing connections, the backend never becomes ready.

**Discriminating evidence:** the db container is unhealthy or restarting too;
backend logs show connection errors. If so this is really RB-003 — go there.

### 4. Disk full

A container that cannot write exits. If RB-001 is also firing, fix that first;
this is a symptom, not the cause.

## Remediation

### Rung 1 — diagnosis only

Identify the container, the exit code, whether the kernel OOM killer fired, and
which process it actually killed. For a memory leak, report the growth rate:
*"RSS climbed 40 MB/hour over six hours"* is the useful finding, and the one
episodic memory makes possible — a fourth OOM kill in three days is a pattern, a
first one is an incident.

### Rung 2 — gated, one named container

```bash
docker restart <lab-container-name>
```

Clears the immediate symptom and nothing else. Only ever on a `lab-*` container.
State plainly in the diagnosis that a restart is not a fix: if the cause is a
leak, the loop returns on the same schedule.

### The permanent fix — human, config change

Give each service a memory limit so a leak is contained to its own container and
the kernel stops picking victims at random:

```yaml
services:
  backend:
    mem_limit: 512m
    mem_reservation: 256m
```

With a limit set, an over-allocating container is OOM-killed *by its own cgroup*,
`{{.State.OOMKilled}}` becomes `true` and meaningful, and `dockerd` stops being a
candidate victim. This one change makes RB-002 diagnosable rather than
speculative — it is the highest-value config change on the host.

## DO NOT

- **Never restart a live stack** to clear a loop: not `docker compose restart`,
  not `docker compose down && up`, on `anglemaze-platform`, `landing-page` or
  `translation-app`. Those serve public traffic. The agent's allowlist rejects
  any target not matching `lab-*`.
- **Never `docker kill` or restart `dockerd`/`containerd`** to "reset" things.
  That stops all six containers at once.
- **Never raise a memory limit above ~1 GiB** on this host. With 3.7 GiB total
  and six containers, an over-generous limit just relocates the OOM.
- **Never disable the restart policy** to stop the noise. That converts a
  self-healing loop into an outage.
- **Never conclude "OOM" from exit code 137 alone.** Confirm with
  `{{.State.OOMKilled}}` or `dmesg`.

## Escalate to a human when

- The kernel killed `dockerd`, `containerd` or `postgres` — that is host-level
  pressure, not one bad container.
- `RestartCount` exceeds 10 in an hour on a live container.
- Memory grows monotonically across restarts — a genuine leak, needing an
  application fix.
- The loop coincides with disk pressure (RB-001) or db problems (RB-003) —
  diagnose the shared cause, not the symptom.
- Swap usage is sustained above ~50%. On 2 vCPUs, swap thrashing makes the box
  unresponsive over SSH, and you may not be able to log in to fix it.

## Reproduce safely

**Only with an explicit memory limit set.** Reproducing an OOM without one means
inviting the host-wide kernel OOM killer to choose a victim among the live
stacks — exactly the outcome the lab boundary exists to prevent.

```yaml
# /home/olav/lab-slice/docker-compose.yml
services:
  lab-victim:
    image: python:3.12-alpine
    container_name: lab-victim
    mem_limit: 128m          # MANDATORY — contains the kill to this container
    restart: unless-stopped
    command:
      - python
      - -u
      - -c
      - |
        import time
        buf = []
        while True:
            buf.append(bytearray(10 * 1024 * 1024))
            time.sleep(1)
    networks: [lab]

networks:
  lab:
```

Allocating 10 MB/s against a 128 MB cap produces a clean cgroup OOM in roughly
13 seconds, then loops on the restart policy. That gives a repeatable incident
for the BDD scenarios, a real `OOMKilled: true` to assert on, and a blast radius
bounded at 128 MB on a host with 3 GB available.
