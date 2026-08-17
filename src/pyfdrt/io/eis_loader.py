"""Shared EIS data-loading utilities.

A single ``load_eis_file`` entry point that understands several measurement file
formats and returns an (N, 3) array of [frequency, Z_real, Z_imag] in a common
convention (Z_imag negative for capacitive points). ``filter_positive_imag``
drops the inductive (Z_imag > 0) points. Used by both the GUI (main_app.py) and
the standalone CLI scripts so every entry point reads data identically.
"""

import re

import numpy as np


def _zview_header_columns(line):
    """Tokenize a ZView column-header line into one name per data column.

    The header is typically a quoted, whitespace-separated line
    (``"  Freq (Hz)    Ampl ... Z'(a)  Z''(b) ..."``) while the data rows are
    comma-separated. A token that is purely a parenthesized unit — ``(Hz)`` —
    is merged into the preceding name, so token positions stay aligned with
    the data columns. A comma-separated header is split on commas instead.
    """
    s = line.strip().strip('"').strip()
    raw = [t.strip() for t in (s.split(',') if ',' in s else s.split())]
    cols = []
    for tok in raw:
        if cols and re.fullmatch(r"\(.*\)", tok):
            cols[-1] += ' ' + tok
        elif tok:
            cols.append(tok)
    return cols


def _parse_zview_z(fname):
    """Parse a ZView/Scribner '.Z' data file.

    Layout: a text header, a single integer line with the point count, a
    column-header line containing 'Freq', then comma-separated data rows.
    Columns are mapped **by name** from that header line via
    ``_classify_column`` — the historical fixed positions 0/4/5 are what the
    standard layout [Freq, Ampl, Bias, Time, Z', Z'', GD, Err, Range] resolves
    to, but an export with a different column set is now read correctly or
    rejected with a clear error instead of loading silently wrong.

    Returns an (N, 3) array of [frequency, Z_real, Z_imag]. The Z'' sign
    convention already matches the plain-text datasets (capacitive < 0), so no
    sign flip is applied for a plain ``Z''`` column; a negated header name
    (``-Z''``) is flipped back like the text loader does.
    """
    with open(fname, 'r', encoding='utf-8-sig') as fh:
        lines = fh.readlines()

    # Data rows start after the column-header line (the one mentioning 'Freq').
    start, header_line = None, None
    for i, line in enumerate(lines):
        if 'Freq' in line:
            start, header_line = i + 1, line
            break
    if start is None:
        raise ValueError("No 'Freq' column header found in .Z file")

    idx, sign = {}, {}
    columns = _zview_header_columns(header_line)
    for j, name in enumerate(columns):
        role, neg = _classify_column(name)
        if role and role not in idx:
            idx[role] = j
            sign[role] = -1.0 if neg else 1.0

    missing = {'freq', 'real', 'imag'} - idx.keys()
    if missing:
        raise ValueError(
            f".Z header does not identify column(s) {sorted(missing)}: {columns}")

    need = max(idx.values())
    rows = []
    for line in lines[start:]:
        parts = line.strip().split(',')
        if len(parts) <= need:
            continue
        try:
            freq = float(parts[idx['freq']]) * sign['freq']
            z_re = float(parts[idx['real']]) * sign['real']
            z_im = float(parts[idx['imag']]) * sign['imag']
        except ValueError:
            continue  # skip blank / non-numeric trailing lines
        rows.append((freq, z_re, z_im))

    if not rows:
        raise ValueError("No numeric data rows found in .Z file")
    return np.array(rows)


def _classify_column(token):
    """Map a header token to ('freq' | 'real' | 'imag' | None, negate?).

    The negate flag is True when the column already stores the negated quantity
    (e.g. '-Z_imag'), so the value must be flipped to recover the internal
    convention of an actual Z_imag (capacitive < 0).
    """
    t = token.strip().lower()
    neg = t.startswith('-')
    core = t.lstrip('+-').strip()
    # Name without a unit suffix or function wrapper, separators removed:
    # 'z_im/ohm' -> 'zim', 'im(z)' -> 'im', '-z_imag [ohm]' -> 'zimag'
    base = re.split(r"[/(\[]", core, maxsplit=1)[0].strip()
    key  = re.sub(r"[^a-z0-9]", "", base)

    if 'freq' in key or key in ('f', 'hz', 'w', 'omega'):
        return 'freq', neg
    # Prime notation: z'' (imag) must be tested before z' (real).
    if core.startswith("z''") or core.startswith('z"'):
        return 'imag', neg
    if core.startswith("z'"):
        return 'real', neg
    # Explicit names only. A bare substring 'im' is NOT enough: it also
    # matches 'time' (a common data column) and silently mis-parses the file.
    if 'imag' in key or key in ('im', 'zi', 'zim', 'imz', 'z2'):
        return 'imag', neg
    if 'real' in key or key in ('re', 'zr', 'zre', 'rez', 'z1', 'z'):
        return 'real', neg
    return None, False


def _parse_headed_text(fname):
    """Parse a text file whose first line names the columns.

    Columns are whitespace-delimited, or comma-delimited when the header line
    contains a comma (task #90, drt-io.md 2.2). Columns are mapped to
    [frequency, Z_real, Z_imag] by their header names (any order), so files
    like 'Z_re  -Z_imag  frequency' are read correctly. A '-Z_imag' (or
    '-Z_re') column is sign-flipped so the returned Z_imag uses the same
    convention as the plain datasets (capacitive < 0).

    In comma mode an EMPTY leading header token - the ',Freq,Real,Imag'
    layout of a pandas DataFrame.to_csv() export - marks the first data
    column as a row index and drops it. Only the empty token triggers the
    drop: a named leading column is kept and left unmatched.
    """
    with open(fname, 'r', encoding='utf-8-sig') as fh:
        lines = fh.readlines()

    header_idx = next(i for i, l in enumerate(lines) if l.strip())
    raw = lines[header_idx].strip()
    comma = ',' in raw
    header = ([t.strip() for t in raw.split(',')] if comma else raw.split())
    offset = 0
    if comma and header and header[0] == '':
        header = header[1:]
        offset = 1          # pandas index column: dropped (drt-io.md 2.2)

    idx, sign = {}, {}
    for j, tok in enumerate(header):
        role, neg = _classify_column(tok)
        if role and role not in idx:
            idx[role] = j
            sign[role] = -1.0 if neg else 1.0

    missing = {'freq', 'real', 'imag'} - idx.keys()
    if missing:
        raise ValueError(
            f"Header does not identify column(s) {sorted(missing)}: {header}")

    data = np.loadtxt(lines[header_idx + 1:], delimiter=',' if comma else None)
    freq = data[:, offset + idx['freq']] * sign['freq']
    z_re = data[:, offset + idx['real']] * sign['real']
    z_im = data[:, offset + idx['imag']] * sign['imag']
    return np.column_stack([freq, z_re, z_im])


def _has_text_header(fname):
    """True if the first non-empty line contains non-numeric column names.

    Tokenized with the shared comma-else-whitespace rule (task #93,
    drt-io.md 2.4), so a purely numeric comma-delimited line is recognised
    as data instead of being mis-read as a single non-numeric header token.
    An EMPTY comma token counts as non-numeric: it is the pandas index-column
    marker of the headed layout (drt-io.md 2.2, task #90), never data.
    """
    with open(fname, 'r', encoding='utf-8-sig') as fh:
        for line in fh:
            if not line.strip():
                continue
            for tok in _split_cols(line.strip()):
                try:
                    float(tok)
                except ValueError:
                    return True
            return False
    return False


def _split_cols(line):
    """Split a data line on commas if present, else on whitespace.

    Comma files (e.g. HIOKI CSV) may have a column name containing a space
    ('AC Status'), so splitting on commas keeps the column indices aligned.
    """
    return line.split(',') if ',' in line else line.split()


def _find_magphase_header(lines):
    """Locate an impedance magnitude / phase column header.

    Returns (row_index, {'freq', 'mag', 'phase': col_index}) for a table that
    provides |Z| and phase (e.g. a HIOKI LCR-meter CSV), or None if the file is
    not in magnitude/phase form.
    """
    for i, line in enumerate(lines):
        cmap = {}
        for j, tok in enumerate(_split_cols(line)):
            core = tok.strip().lower().lstrip('+-').strip()
            if 'freq' in core and 'freq' not in cmap:
                cmap['freq'] = j
            elif core in ('z', '|z|', 'mag', 'magnitude', 'absz') and 'mag' not in cmap:
                cmap['mag'] = j
            elif (core in ('ph', 'phase', 'theta', 'deg') or 'phase' in core) \
                    and 'phase' not in cmap:
                cmap['phase'] = j
        if {'freq', 'mag', 'phase'} <= cmap.keys():
            return i, cmap
    return None


def _parse_magnitude_phase(fname):
    """Parse a |Z| / phase table into [frequency, Z_real, Z_imag].

    Z_real = |Z|*cos(phase), Z_imag = |Z|*sin(phase). Phase is treated as
    degrees unless every value lies within +/-pi (then assumed radians). The
    resulting Z_imag is negative for capacitive points, matching the other
    loaders.
    """
    with open(fname, 'r', encoding='utf-8-sig') as fh:
        lines = fh.readlines()
    found = _find_magphase_header(lines)
    if found is None:
        raise ValueError("No frequency / |Z| / phase header found")
    hdr_i, cmap = found
    need = max(cmap.values())

    rows = []
    for line in lines[hdr_i + 1:]:
        parts = _split_cols(line)
        if len(parts) <= need:
            continue
        try:
            f   = float(parts[cmap['freq']])
            mag = float(parts[cmap['mag']])
            ph  = float(parts[cmap['phase']])
        except ValueError:
            continue
        rows.append((f, mag, ph))

    if not rows:
        raise ValueError("No numeric data rows found")
    arr = np.array(rows)
    ph = arr[:, 2]
    if np.nanmax(np.abs(ph)) > np.pi:   # out of radian range -> phase in degrees
        ph = np.deg2rad(ph)
    z_re = arr[:, 1] * np.cos(ph)
    z_im = arr[:, 1] * np.sin(ph)
    return np.column_stack([arr[:, 0], z_re, z_im])


def _validate_monotonic_freq(arr, fname):
    """Reject a non-monotonic frequency column with a clear error (task #31).

    Downstream sweep-direction handling (``ensure_ascending_2d``) compares only
    the endpoints, so a file whose frequency column is neither strictly
    ascending nor strictly descending would silently mis-sort and yield
    meaningless results. Every load path funnels through this check, so arrays
    leaving the loader are guaranteed strictly monotonic.
    """
    if arr.ndim != 2 or arr.shape[0] < 2:
        return arr
    freq = arr[:, 0]
    df = np.diff(freq)
    if (df > 0).all() or (df < 0).all():
        return arr
    # Locate the first break relative to the initial direction for the message.
    if df[0] == 0:
        bad = 1
    else:
        offend = np.nonzero(df <= 0 if df[0] > 0 else df >= 0)[0]
        bad = int(offend[0]) + 1
    raise ValueError(
        f"Non-monotonic frequency column in '{fname}': frequencies must be "
        f"strictly ascending or strictly descending, but data row {bad} "
        f"(f = {freq[bad]:g}) breaks the sweep direction.")


def load_eis_file(fname):
    """Load an EIS file as an (N, 3) array of [frequency, Z_real, Z_imag].

    Supports ZView '.Z' files, magnitude/phase tables (e.g. HIOKI LCR CSV),
    headerless text in [freq, re, im] order, and header-labelled text whose
    columns are mapped by name (any order).

    Regardless of format, the frequency column of the loaded array is validated
    to be strictly monotonic; a shuffled or duplicated frequency sweep raises a
    ``ValueError`` instead of loading silently.
    """
    if fname.lower().endswith('.z'):
        return _validate_monotonic_freq(_parse_zview_z(fname), fname)
    with open(fname, 'r', encoding='utf-8-sig') as fh:
        lines = fh.readlines()
    if _find_magphase_header(lines) is not None:
        return _validate_monotonic_freq(_parse_magnitude_phase(fname), fname)
    if _has_text_header(fname):
        return _validate_monotonic_freq(_parse_headed_text(fname), fname)
    # Headerless [f, Z_re, Z_im] table, whitespace- or comma-delimited
    # (task #93, drt-io.md 2.4).
    first = next((l for l in lines if l.strip()), '')
    delim = ',' if ',' in first else None
    return _validate_monotonic_freq(np.loadtxt(fname, delimiter=delim), fname)


def filter_positive_imag(data):
    """Drop points with a positive imaginary part (inductive tail).

    Keeps only capacitive / resistive points (Z_imag <= 0). Returns
    (filtered_data, n_dropped).
    """
    mask = data[:, 2] <= 0
    return data[mask], int((~mask).sum())


# Over-correction spike trim (drt-preprocessing.md 2.4.1, task #91). At the
# top of the sweep omega*L dominates |Z_im|, so a few-percent error in the
# fitted L leaves large residuals after subtraction: positive ones fall to
# the sign filter above, but NEGATIVE ones survive it as spurious capacitive
# spikes at the HF edge. An edge point inside the originally-inductive region
# is dropped when its |Z_im| exceeds SPIKE_TRIM_FACTOR times the median of
# the next SPIKE_TRIM_WINDOW kept points; the walk stops at the first sane
# point. Threshold sits between the measured genuine spikes (6.2-15.4x) and
# the SOFC borderline (4.1x, removal shows no benefit).
SPIKE_TRIM_FACTOR = 5.0
SPIKE_TRIM_WINDOW = 5


def trim_overcorrected_hf(data, raw_data):
    """Drop over-corrected capacitive spikes at the high-frequency edge.

    ``data`` is the L-corrected, sign-filtered array; ``raw_data`` the
    pre-correction array (used only to bound the originally-inductive
    frequency region). Only edge points inside that region are candidates,
    and the walk from the highest kept frequency stops at the first
    non-flagged point (edge trim, never an interior outlier pass). Returns
    (trimmed_data, trimmed_frequencies) with the input row order preserved.
    """
    inductive = raw_data[:, 2] > 0
    if not inductive.any() or len(data) == 0:
        return data, []
    f_ind_min = float(raw_data[inductive, 0].min())

    order = np.argsort(data[:, 0])
    e = data[order]
    trimmed = []
    while len(e) > 1:
        f, zim = e[-1, 0], e[-1, 2]
        if f < f_ind_min:
            break
        ref = float(np.median(np.abs(
            e[max(0, len(e) - 1 - SPIKE_TRIM_WINDOW):-1, 2])))
        if ref > 0 and abs(zim) > SPIKE_TRIM_FACTOR * ref:
            trimmed.append(float(f))
            e = e[:-1]
        else:
            break
    if not trimmed:
        return data, []
    keep = ~np.isin(data[:, 0], trimmed)
    return data[keep], trimmed
