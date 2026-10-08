"""Example: scale-invariance sweep for SI-SDR-style losses.

A 1-D output sweep cannot capture scale-invariant waveform losses, so this
example sweeps an *attenuation factor* gamma: pred = gamma * signal (+ noise).

Run:  python examples/scale_invariance.py
"""

import numpy as np


def si_sdr_dB(pred: np.ndarray, ref: np.ndarray, eps: float = 1e-8) -> float:
    p = pred - pred.mean()
    r = ref - ref.mean()
    alpha = float(np.dot(p, r) / (np.dot(r, r) + eps))
    s_target = alpha * r
    e = p - s_target
    val = 10 * np.log10((np.dot(s_target, s_target) + eps) / (np.dot(e, e) + eps))
    return float(np.clip(val, -30, 30))


def main():
    rng = np.random.default_rng(0)
    s = rng.standard_normal(16000)
    noise = rng.standard_normal(16000)

    print("=== SI-SDR vs attenuation factor gamma (pred = gamma*s) ===")
    print("Scale invariance: for gamma>0 the loss is identical (gamma^2 cancels).")
    for gamma in [0.0, 1e-3, 0.1, 0.5, 1.0, 2.0, 10.0]:
        pred = gamma * s
        sd = si_sdr_dB(pred, s)
        print(f"  gamma={gamma:6.3f}  SI-SDR={sd:8.2f}dB  loss(-sisdr)={-sd:8.2f}")

    print("\n=== SI-SDR vs added noise epsilon (pred = s + eps*noise) ===")
    print("Phase sensitivity: time-domain inner product penalizes any noise direction.")
    for eps in [0.0, 0.1, 0.3, 0.5, 1.0, 2.0]:
        pred = s + eps * noise
        sd = si_sdr_dB(pred, s)
        print(f"  eps={eps:5.2f}  SI-SDR={sd:8.2f}dB  loss(-sisdr)={-sd:8.2f}")


if __name__ == "__main__":
    main()
