"""sea_g2p.bin → Vietnamese(+English) only: same merged/common tables, no Thai/Indonesian sections.
Verified byte-identical phonemes vs the full dictionary."""
import sys
from pathlib import Path
src, scripts, dst = sys.argv[1:4]
sys.path.insert(0, scripts)
from seap import load_bin, write_bin_v2
merged, common, _ = load_bin(Path(src))
write_bin_v2(Path(dst), merged, common, {})
