#!/bin/bash
echo "ulimit -t (soft):  $(ulimit -t)"
echo "ulimit -Ht (hard): $(ulimit -Ht)"
python -c "import resource; s,h=resource.getrlimit(resource.RLIMIT_CPU); print(f'py soft={s} hard={h}')"
ulimit -t unlimited 2>&1 && echo "raised OK" || echo "raise failed"
echo "after raise: $(ulimit -t)"
# try via prlimit
echo "---prlimit---"
prlimit -p $$ --cpu 2>&1 || true
