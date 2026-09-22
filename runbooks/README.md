# Runbook corpus — `devops-platform`

Operational procedures for the Docker Host Triage Agent. This directory is
**procedural memory**: it tells the agent (and a human on call) *what to do here*,
in what order, and what is forbidden.

It is **not** the same corpus as `../../docs/`. Those are agentic-AI textbooks for
the team. Nothing in `../../docs/` belongs in the retrieval index.

## Conventions

Every runbook uses the same section order, and each section is written to stand
alone so it survives chunking:

| Section | Purpose |
|---|---|
| YAML frontmatter | Retrieval filters + dispatcher metadata |
| **Symptom** | How the alert fires; what the operator sees |
| **Confirm** | Read-only checks with expected output. Rung 1 only. |
| **Likely causes** | Ordered by frequency *on this host*, each with discriminating evidence |
| **Remediation** | Per cause, tagged by autonomy rung |
| **DO NOT** | Hard prohibitions. The highest-value section. |
| **Escalate** | Conditions requiring a human |
| **Reproduce safely** | How to trigger this on the lab slice for tests and demos |

Chunk on `##` headings. Keep **DO NOT** and **Remediation** in the same chunk as
their parent cause where possible — a retrieved fix without its prohibition is
worse than no retrieval.

## Autonomy rungs

| Rung | Meaning | Agent may |
|---|---|---|
| 1 | Read-only observation | run any command in **Confirm** unattended |
| 2 | Gated remediation | propose a **Remediation** action; executes only after typed human approval; always writes to the audit log first |
| 3 | Bounded auto-remediation | **not in scope for this project** |

## The lab/live boundary

The VPS runs three live stacks serving public traffic:
`anglemaze-platform` (proxy, frontend, backend, db), `landing-page`,
`translation-lab`.

**The agent's dispatcher allowlist permits write actions only on containers whose
name starts with `lab-`.** On live containers the agent is rung 1 — it may look,
diagnose and recommend, but the allowlist rejects every write tool. That
restriction lives in the dispatcher, not in a prompt, so no model output can
bypass it.

Proposed lab slice: `lab-victim` (a container we break on purpose), `lab-db`
(throwaway Postgres), on network `lab`, defined in `/home/olav/lab-slice/`.
This does not exist yet — building it is Sprint 1 work.

## Host facts these runbooks assume

Ubuntu 24.04.4, kernel 6.8, Hetzner KVM. **2 vCPU · 3.7 GiB RAM · 2 GiB swap ·
75 GB disk (6.2 GB used at snapshot).** Docker 29.4.2, Compose v5.1.3.
Stacks live under `/home/olav/`. Sole admin user `olav` (sudo + docker).
`ufw` allows 22, 80, 443 inbound only.

**Two latent problems, true of every stack at snapshot time:**

1. **No logging limits.** No compose file sets a `logging:` block and there is no
   `/etc/docker/daemon.json` log config, so all containers use `json-file` with
   no `max-size` or `max-file`. Container logs grow without bound. See RB-001.
2. **No memory limits.** No compose file sets `mem_limit` or
   `deploy.resources.limits`, so `docker stats` reports the host total (3.725 GiB)
   as every container's limit. There is no per-container cgroup ceiling, which
   means the *kernel* OOM killer chooses the victim by score — and it may kill
   `dockerd` or `postgres` rather than the process actually leaking. See RB-002.

## Index

| ID | Incident | Rung reachable |
|---|---|---|
| [RB-001](RB-001-disk-usage-high.md) | Disk usage high on `/` | 2 |
| [RB-002](RB-002-container-restart-loop.md) | Container restart loop / OOM kill | 2 |
| [RB-003](RB-003-postgres-connection-exhaustion.md) | Postgres connection exhaustion | 2 |
