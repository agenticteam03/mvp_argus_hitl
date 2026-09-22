# what: the image that runs both the exporter and the agent.
# why:  on the laptop these were host processes; on the VM they have to be
#       containers, because nothing else on that machine is allowed to grow a
#       Python virtualenv and a scikit-learn install. One image for both, because
#       they share every dependency and differ only in which module is started.
# how:  install the requirements, copy the code, and set no CMD worth arguing
#       about -- docker-compose.yml names the module for each service.

# slim (Debian, glibc), NOT alpine (musl), and the reason is architecture. The
# target VM is ARM64. scikit-learn publishes manylinux aarch64 wheels but no
# musllinux aarch64 wheel, so on alpine pip would fall back to compiling it from
# source -- on one vCPU, that is a very long way to discover a decision you
# could have made here in one word.
FROM python:3.12-slim

# PYTHONUNBUFFERED: without it Python block-buffers stdout when it is a pipe, so
# `docker logs` shows nothing for minutes and then everything at once. In a
# container the logs ARE the interface, so the buffer is a liability.
# PYTHONDONTWRITEBYTECODE: __pycache__ in a layer is dead weight.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# requirements.txt is copied and installed BEFORE the source. Docker caches each
# layer by the files it was built from, so editing agent/graph.py then rebuilds
# only the last two layers instead of reinstalling scikit-learn. On a 1-vCPU ARM
# VM that is the difference between a 4-second rebuild and a 4-minute one.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The code, the runbooks, and the tests.
#
# runbooks/ is not documentation here: it is the agent's PROCEDURAL MEMORY, the
# corpus TfidfDocs reads at start-up. Leave it out and Stage 4 dies with "No
# readable markdown under runbooks/...".
#
# tests/ is in the image on purpose, which is not the usual advice. The reason
# is that the deployment target has no Python and no virtualenv, so without this
# there is no way to answer "did the checkout arrive intact?" on the VM itself
# before spending an afternoon debugging Docker networking. 113 offline tests
# are a few hundred kilobytes and they run in six seconds.
COPY agent/ ./agent/
COPY runbooks/ ./runbooks/
COPY tests/ ./tests/

# Overridden per service in docker-compose.yml. Named here so that a bare
# `docker run` of this image does something meaningful instead of nothing.
CMD ["python", "-m", "agent.run"]
