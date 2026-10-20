"""Run a command and record its wall time and peak GPU memory (NVML, the value nvidia-smi shows, summed over the
process tree and all GPUs, sampled every 0.1 s). Copy of thread0 repro/eval/gpu_peak.py.

usage: python gpu_peak.py <out.json> -- <command ...>
"""
import json, os, subprocess, sys, threading, time

import psutil
import pynvml


def main():
    out = sys.argv[1]
    cmd = sys.argv[sys.argv.index("--") + 1:]
    pynvml.nvmlInit()
    hs = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
    peak, peak_gpu, stop = [0], {}, threading.Event()
    t0 = time.time()
    proc = subprocess.Popen(cmd)
    ps = psutil.Process(proc.pid)

    def sample():
        while not stop.is_set():
            try:
                own = {ps.pid} | {c.pid for c in ps.children(recursive=True)}
            except psutil.NoSuchProcess:
                own = set()
            tot = 0
            for i, h in enumerate(hs):
                m = sum(p.usedGpuMemory or 0 for p in pynvml.nvmlDeviceGetComputeRunningProcesses(h) if p.pid in own)
                if m:
                    peak_gpu[i] = max(peak_gpu.get(i, 0), m)
                tot += m
            peak[0] = max(peak[0], tot)
            time.sleep(0.1)

    th = threading.Thread(target=sample, daemon=True)
    th.start()
    rc = proc.wait()
    wall = time.time() - t0
    stop.set()
    th.join()
    name = lambda i: (lambda n: n.decode() if isinstance(n, bytes) else n)(pynvml.nvmlDeviceGetName(hs[i]))
    res = {"cmd": cmd, "rc": rc, "wall_s": round(wall, 1), "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
           "nvml_peak_GiB": round(peak[0] / 2 ** 30, 3), "by_gpu": {name(i): round(v / 2 ** 30, 3) for i, v in peak_gpu.items()},
           "driver": pynvml.nvmlSystemGetDriverVersion()}
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res), flush=True)
    sys.exit(rc)


if __name__ == "__main__":
    main()
