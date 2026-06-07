# %% [markdown]
# # Absolute vs relative phase reference (S-transform vs wavelet)
#
# A time-frequency coefficient is magnitude·e^{iφ}. The magnitude is "how much
# of frequency f near time τ"; the phase φ is measured against a reference. Two
# choices:
#
# - **absolute** (STFT / S-transform): kernel `e^{-i2πf·t}`, locked to the global
#   origin t=0. Phase is referenced to absolute time.
# - **relative** (CWT / wavelet): kernel `e^{-i2πf·(t-τ)}`, referenced to the
#   wavelet's own moving centre τ.
#
# They differ ONLY by `t` vs `(t-τ)` in the exponent, i.e. by a unit-modulus
# ramp `e^{i2πf·τ}`:  W(τ,f) = e^{i2πf·τ} · S(τ,f).  So the MAGNITUDE is identical;
# only the PHASE differs. This script analyses a steady tone with both (same
# Gaussian window) and shows: magnitude identical, absolute phase FLAT, relative
# phase a winding RAMP — and that undoing the ramp turns one into the other.

# %%
import numpy as np
import matplotlib.pyplot as plt

N = 512
t = np.arange(N)
f0 = 64                      # a steady tone at bin 64
x = np.cos(2 * np.pi * f0 * t / N)

# A Gaussian window centred at each τ (constant width here, for clarity).
sigma = 24.0


def coeffs_at(f):
    """Absolute- and relative-reference coefficients along τ at frequency f."""
    S = np.empty(N, complex)   # absolute (STFT / S-transform):  e^{-i2πf t}
    W = np.empty(N, complex)   # relative (wavelet):             e^{-i2πf (t-τ)}
    for tau in range(N):
        w = np.exp(-((t - tau) ** 2) / (2 * sigma**2))
        S[tau] = np.sum(x * w * np.exp(-1j * 2 * np.pi * f * t / N))
        W[tau] = np.sum(x * w * np.exp(-1j * 2 * np.pi * f * (t - tau) / N))
    return S, W


S, W = coeffs_at(f0)

print(f"max |‖S‖ − ‖W‖|  = {np.max(np.abs(np.abs(S) - np.abs(W))):.2e}   (magnitudes identical)")
print(f"W / S  ==  e^(i2πf·τ)?  max err = "
      f"{np.max(np.abs(W / S - np.exp(1j*2*np.pi*f0*t/N))):.2e}")
# 'unwind' the relative phase by the known ramp -> recovers the absolute phase
W_corrected = W * np.exp(-1j * 2 * np.pi * f0 * t / N)
print(f"relative phase after removing the 2πfτ ramp == absolute phase? max err = "
      f"{np.max(np.abs(np.angle(W_corrected) - np.angle(S))):.2e}")

# %%
fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
a1.plot(t, np.abs(S), label="|absolute|", lw=2)
a1.plot(t, np.abs(W), "--", label="|relative|")
a1.set_ylabel("magnitude")
a1.set_title(f"Steady tone f={f0}: magnitude is identical for both references")
a1.legend()

a2.plot(t, np.angle(S), label="absolute (S-transform): FLAT")
a2.plot(t, np.angle(W), label="relative (wavelet): winding RAMP")
a2.set_ylabel("phase (rad)")
a2.set_title("Phase along time — the whole difference")
a2.legend()

a3.plot(t, np.unwrap(np.angle(W)), label="relative, unwrapped (slope = 2πf₀/N)")
a3.plot(t, np.unwrap(np.angle(W_corrected)), "--", label="relative minus 2πfτ ramp = absolute")
a3.set_ylabel("unwrapped phase")
a3.set_xlabel("time τ (samples)")
a3.set_title("Removing the e^{i2πfτ} ramp converts relative → absolute")
a3.legend()
fig.tight_layout()
plt.show()
