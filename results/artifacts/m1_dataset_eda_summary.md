# M1 Dataset EDA Summary

The Suk coronary mesh dataset is present in the expected steady-flow layout.
Raw HDF5 files remain outside version control; this artifact records reproducible metadata.

| Subset | Samples | Nodes Mean | Nodes Range | Faces Mean | WSS Magnitude Mean | Max WSS Magnitude | Pressure Range Mean | MD5 |
|---|---:|---:|---|---:|---:|---:|---:|---|
| single | 2000 | 9188.85 | 5466-12160 | 18373.69 | 29.1146 | 548.8169 | 5215.1119 | `ba365decba2357fb7b24de641a2133a1` |
| bifurcating | 1999 | 16836.59 | 8600-24800 | 33669.17 | 13.0598 | 323.1414 | 736.9501 | `b73d96148e4245be1121d57efb6e3d63` |

Report use:

- Use sample counts and checksums in the data section.
- Use node and face ranges to justify mesh-size handling.
- Use WSS and pressure summaries to sanity-check target scale before training.
