# llmbench summary-score snapshot

Date: 2026-11-17.

Stark now uses llmbench's own published coding and non-coding summary scores for
model browsing and Beginner recommendations. The application retains the
scored parent repository only; quantized exports inherit a score solely through
the existing exact review, declared-parent or unambiguous-name binding rules.

## Source binding

- Site generation: `2026-11-17T21:02:32Z`
- Summary CSV:
  `https://llmbench.site/downloads/llmbench-summary-scores.csv`
- Summary CSV SHA-256:
  `16f1757328eb8477daf2d81aa0e1ae78fc23a7e580df803ad6bae2420e512665`
- Model catalogue CSV:
  `https://llmbench.site/downloads/llmbench-models.csv`
- Model catalogue CSV SHA-256:
  `9fa830d4ab41feab7721fb3258b53ec179962f6c21a3604a60dde4e3adcaa9ae`
- Method: `https://llmbench.site/summary/`
- Data and source licences: `https://llmbench.site/data/`

The packaged open-model subset has 92 model rows: 88 carry coding scores and
45 carry non-coding scores. Missing values remain unknown; one task's score is
never substituted for the other.

This snapshot changes recommendation and browsing evidence only. It does not
qualify any checkpoint, quantization, native backend, image path, MTP setting
or installed build.
