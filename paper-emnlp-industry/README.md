# paper-emnlp-industry/

EMNLP 2026 Industry Track submission workspace, parallel to
[`../paper/`](../paper/) (TKDE version). The TKDE folder is NOT
touched by anything in here.

## Status

| Item                                        | Status      |
|---------------------------------------------|-------------|
| Outline (page-by-page layout + appendix map)| Done — [OUTLINE.md](OUTLINE.md) |
| Abstract (Draft A selected)                 | Done — [ABSTRACT.md](ABSTRACT.md), embedded in [main.tex](main.tex) |
| `main.tex` skeleton                         | Done        |
| Section stubs (1-9 + appendices A-E)        | Done (TODOs inside each file) |
| ACL style files (`acl_latex.sty` etc.)      | NOT downloaded yet — get from `acl-org/acl-style-files` |
| Section content (rewrite from TKDE source)  | Not started |
| Figures (anonymise + recompile)             | Not started |
| `references.bib` (subset of TKDE refs)      | Not started |
| Anonymisation pass                          | Not started |

## How to compile (once acl_latex.sty is in place)

```
cd paper-emnlp-industry
# Drop acl_latex.sty + acl_natbib.bst into this directory first
latexmk -pdf main.tex
```

## Folder layout

```
paper-emnlp-industry/
├── README.md              <- you are here
├── OUTLINE.md             <- 6-page layout + main-text/appendix mapping
├── ABSTRACT.md            <- Draft A (final) + Draft B (reference)
├── main.tex               <- ACL template skeleton with abstract embedded
├── references.bib         <- (TODO: subset of ../paper/reference.bib)
└── sections/
    ├── 1-intro.tex
    ├── 2-related.tex
    ├── 3-pipeline.tex
    ├── 4-bench.tex
    ├── 5-results.tex
    ├── 6-lessons.tex       <- NEW (not in TKDE version)
    ├── 7-conclusion.tex
    ├── 8-limitations.tex   <- REQUIRED, outside page budget
    ├── 9-ethics.tex        <- outside page budget
    ├── A-fm-notation.tex
    ├── B-pipeline-math.tex
    ├── C-bench-details.tex
    ├── D-reference-methods.tex
    └── E-full-tables.tex
```

## Mapping back to TKDE sources

Each section stub names the TKDE source file(s) it should compress
or adapt. See [OUTLINE.md](OUTLINE.md) for the full
main-text → appendix mapping table.

## Anonymisation

`main.tex` declares four placeholder system names so the entire
submission stays anonymous without us having to scan section files:

| Submission name      | Real name (camera-ready) |
|----------------------|--------------------------|
| `\sysname`           | DeXposure-Claw           |
| `\fmname`            | DeXposure-FM             |
| `\benchname`         | DeXposure-Bench          |
| `\corpusname`        | DeXposure dataset        |

Flip these in `main.tex` for the camera-ready version.

See [OUTLINE.md](OUTLINE.md) §"Anonymisation Checklist" for the full
pre-submission scan list (author block, self-citations, figure PDF
exif, code-availability statement, etc.).

## Open decisions before submission

1. Are `DeXposure-*` names already public (GitHub / arXiv preprint)?
   If yes, keep placeholders through the submission round.
2. Add a second LLM (e.g. GPT-4o) for headline-result breadth?
   Industry reviewers commonly ask this.
3. Code release: anonymous GitHub vs 4open.science vs zip upload?

## Next-step shortlist

In priority order:

1. Confirm the 3 open decisions above.
2. Write [sections/1-intro.tex](sections/1-intro.tex) (NLP-framing intro).
3. Build the headline consolidated results table for Section 5.
4. Compress Algorithm 1 + equations into Appendix B.
5. Pull `acl_latex.sty` + sanity-compile the skeleton.
6. Build `references.bib` subset.
