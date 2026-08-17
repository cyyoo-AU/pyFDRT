# pyFDRT — distribution

This repository is the **distribution** of pyFDRT: the shipped product and
nothing else. It is generated from an upstream working tree, one path at a
time, from a fixed list. **Nothing is authored here.**

The specifications, the golden regression corpus that pins the numerical
behaviour, the test suite and CI are maintained upstream and are not part of
this distribution. `README.md` says how to obtain them.

## Before you change anything here

- **Edits made here do not survive.** Each release replaces the listed paths
  wholesale, so a fix committed here is overwritten by the next one and never
  reaches the source it was meant to fix. Send it upstream instead — the
  contact address is in `README.md`.
- **Do not merge or pull between this repository and an upstream clone.** Git
  refuses one anyway — the two histories are unrelated — and that refusal is a
  guard rather than an obstacle, so do not reach for
  `--allow-unrelated-histories` to get past it. A merge adds files that this
  repository deliberately does not carry. Releases are produced by replacing
  listed paths, not by merging.
- **Do not add files.** The file count is one of the release checks; an extra
  file fails it.
- **Nothing can be validated here.** There is no test suite and no CI in this
  repository, so a change made here cannot be checked even in principle.

## Why some comments cite documents that are not here

Comments in `src/` and `scripts/` cite specification and reference documents
that live upstream. Those citations are deliberately left as they are:
rewriting them would fork the source between the two trees, so a reader would
no longer be able to tell whether a comment describes the code that shipped.
Do not "fix" them.
