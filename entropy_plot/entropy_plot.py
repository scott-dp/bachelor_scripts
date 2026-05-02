import math
from collections import Counter
import matplotlib.pyplot as plt

PATH = "../../bin/SCE6010/SCE6010-D.3.2.0.dat"   # change if needed
WIN = 2560     # window size
STEP = 16     # stride

data = open(PATH, "rb").read()

def entropy(chunk: bytes) -> float:
    n = len(chunk)
    c = Counter(chunk)
    ent = 0.0
    for k in c.values():
        p = k / n
        ent -= p * math.log2(p)
    return ent

offsets = []
entropies = []

for off in range(0, len(data) - WIN, STEP):
    e = entropy(data[off:off + WIN])
    offsets.append(off)
    entropies.append(e)

# ---- PLOT ----
plt.figure(figsize=(12, 5))
plt.plot(offsets, entropies, linewidth=1)
plt.xlabel("Byte Offset")
plt.ylabel("Entropy (bits)")
plt.title("Sliding-Window Entropy Analysis")
plt.grid(True)
plt.tight_layout()
plt.show()
