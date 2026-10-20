"""train.py with one extra Lightning callback: every --every optimizer steps it prints a JSON line with the real
training speed (step/s over the last interval), the logged losses and the peak GPU memory (ROOT/AGENTS.md 3.4), and
at every validation end the validation losses. Everything else is train.py unchanged.

usage (cwd = le-wm checkout): python acwm/train_timed.py [hydra overrides ...]   (STEP_LOG_EVERY=100 by default)
"""
import json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lightning as pl  # noqa: E402
import torch  # noqa: E402


def _num(v):
    try:
        return float(v)
    except Exception:
        return None


class StepLog(pl.Callback):
    def __init__(self, every):
        self.every, self.t, self.n = every, None, 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if self.t is None:
            torch.cuda.synchronize()
            self.t, self.n = time.time(), step
            return
        if step - self.n >= self.every:
            torch.cuda.synchronize()
            now = time.time()
            m = {k: _num(v) for k, v in trainer.callback_metrics.items() if k.startswith("train/")}
            print("STEPLOG " + json.dumps({"step": step, "epoch": trainer.current_epoch, "step_per_s": (step - self.n) / (now - self.t),
                                           "lr": trainer.optimizers[0].param_groups[0]["lr"], **m,
                                           "peak_alloc_GiB": torch.cuda.max_memory_allocated() / 2**30,
                                           "peak_reserved_GiB": torch.cuda.max_memory_reserved() / 2**30,
                                           "utc": time.strftime("%H:%M:%S", time.gmtime(now))}), flush=True)
            self.t, self.n = now, step

    def on_validation_epoch_end(self, trainer, pl_module):
        m = {k: _num(v) for k, v in trainer.callback_metrics.items() if not k.startswith("train/")}
        print("VALLOG " + json.dumps({"step": trainer.global_step, "epoch": trainer.current_epoch, **m}), flush=True)
        self.t = None  # validation time is not training speed


_Trainer = pl.Trainer


class Trainer(_Trainer):
    def __init__(self, *a, callbacks=None, **k):
        super().__init__(*a, callbacks=list(callbacks or []) + [StepLog(int(os.environ.get("STEP_LOG_EVERY", 100)))], **k)


pl.Trainer = Trainer

if __name__ == "__main__":
    import train  # noqa: E402

    sys.argv[0] = train.__file__
    train.run()
