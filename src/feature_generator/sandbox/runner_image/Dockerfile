# Minimal sandbox image for executing already-statically-vetted,
# LLM-generated feature modules. Docker isolation (--network none, resource
# limits, non-root, read-only rootfs -- all applied by the host in
# docker_runner.py) is tier-2 defense-in-depth; tier 1 is the AST-based
# sandbox.static_checks pass that must already have succeeded before any
# module ever reaches this image.
FROM python:3.11-slim

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

RUN useradd -m -u 1000 sandboxuser
COPY runner.py /sandbox/runner.py

USER sandboxuser
WORKDIR /sandbox
ENTRYPOINT ["python", "/sandbox/runner.py"]
