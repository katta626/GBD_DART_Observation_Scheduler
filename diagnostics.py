"""Diagnostic plots for the DIAGNOSTIC PLOTS tab, read directly from the PDR HDF5 files.

Draws the four plots of the original scripts in live_PDR_plot.zip (pearson_CC,
cc_tseries_phase, spectrogram, one_hr_plot) as interactive Plotly charts.

The auto- and cross-correlations are already powers: p_sum_avg = acorr1 + acorr2 + 2*Re(corr)
to within 0.1% in the real data. So power in dB is 10*log10(|x|), not the original
10*log10(|x|^2), and band-averaged correlated power is |mean(corr)|, not its square.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# The acquisition leaves 96-byte HDF5 files (a header with no data) next to the real ones.
MIN_VALID_FILE_BYTES = 1024
# More columns than a screen has pixels only slows the browser; one hour is ~7000 spectra.
MAX_SPECTROGRAM_COLUMNS = 720

SPECTRUM_DATASETS = ("timestamp", "corr_spec_avg", "acorr_spec_avg1", "acorr_spec_avg2")
WINDOW_DATASETS = ("timestamp", "corr_spec_avg", "p_sum_avg")

# Files that are empty, truncated, or still being written.
UNREADABLE_FILE_ERRORS = (OSError, KeyError, ValueError)

# Chart palette for the dark plot surface, checked with the dataviz palette validator.
SURFACE = "#0c1c34"
TEXT_PRIMARY = "#f0efec"
TEXT_SECONDARY = "#c3c2b7"
GRID = "rgba(195, 194, 183, 0.16)"
SERIES_1 = "#3987e5"
SERIES_2 = "#d95926"
HEATMAP_SCALE = [[0.0, "#0d366b"], [0.25, "#1c5cab"], [0.5, "#3987e5"], [0.75, "#86b6ef"], [1.0, "#cde2fb"]]


@dataclass(frozen=True)
class Band:
    center_mhz: float = 175.0
    bandwidth_mhz: float = 16.5
    channel0_is_high: bool = True

    def channel_frequencies(self, channel_count: int) -> np.ndarray:
        """Centre frequency (MHz) of every channel, in channel order, as DAT_FREQ in the FITS files."""
        width = self.bandwidth_mhz / channel_count
        ascending = self.center_mhz - self.bandwidth_mhz / 2 + (np.arange(channel_count) + 0.5) * width
        return ascending[::-1] if self.channel0_is_high else ascending


# ------------------------------------------------------------------------------
# Reading files
# ------------------------------------------------------------------------------
def list_candidate_files(directory: Path) -> tuple[list[Path], int]:
    """HDF5 files oldest to newest, and how many empty placeholder files were skipped."""
    candidates = []
    skipped = 0
    for file_path in Path(directory).glob("*.hdf5"):
        try:
            stat = file_path.stat()
        except OSError:
            # The sync process may prune an old file while this runs.
            continue
        if stat.st_size < MIN_VALID_FILE_BYTES:
            skipped += 1
            continue
        candidates.append((stat.st_mtime, file_path.name, file_path))
    return [entry[2] for entry in sorted(candidates)], skipped


def load_arrays(file_path: Path, dataset_names: tuple[str, ...]) -> dict[str, np.ndarray]:
    import h5py

    with h5py.File(file_path, "r", locking=False) as hdf5_file:
        arrays = {name: np.asarray(hdf5_file[name][:]) for name in dataset_names}
    if arrays["timestamp"].size == 0:
        raise ValueError(f"{Path(file_path).name} contains no spectra")
    return arrays


def load_phase_rotation(calibration_path: Path, channel_count: int) -> np.ndarray:
    """rotate = 1j**(phase_input/90) from rf_path_phase.dat, as in the original script."""
    phase_input = np.loadtxt(calibration_path, dtype=float, delimiter=",")
    if phase_input.size != channel_count:
        raise ValueError(f"expected {channel_count} channels, found {phase_input.size}")
    return 1j ** (phase_input / 90)


def collect_window(
    files: list[Path], end_timestamp: float, window_seconds: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Samples from the window ending at end_timestamp; replaces the old combined HDF5 file."""
    cutoff = end_timestamp - window_seconds
    timestamps, correlation, pa_power = [], [], []
    for file_path in reversed(files):
        try:
            arrays = load_arrays(file_path, WINDOW_DATASETS)
        except UNREADABLE_FILE_ERRORS:
            continue
        file_times = arrays["timestamp"]
        selected = (file_times >= cutoff) & (file_times <= end_timestamp)
        if np.any(selected):
            timestamps.append(file_times[selected])
            correlation.append(arrays["corr_spec_avg"][selected])
            pa_power.append(arrays["p_sum_avg"][selected])
        elif file_times.max() < cutoff:
            break

    if not timestamps:
        return np.array([]), np.empty((0, 0)), np.empty((0, 0))
    all_times = np.concatenate(timestamps)
    order = np.argsort(all_times)
    return all_times[order], np.concatenate(correlation)[order], np.concatenate(pa_power)[order]


# ------------------------------------------------------------------------------
# Calculations
# ------------------------------------------------------------------------------
def power_db(values: np.ndarray) -> np.ndarray:
    """10*log10(|x|) of a power, with a floor so an exact zero does not give -inf."""
    return 10 * np.log10(np.maximum(np.abs(values), np.finfo(float).tiny))


def pearson_magnitude(corr: np.ndarray, acorr1: np.ndarray, acorr2: np.ndarray) -> np.ndarray:
    """|rho| = |corr| / sqrt(|acorr1| * |acorr2|), from 0 to 1.

    The original plotted the complex ratio, and plt.plot() silently kept only its real part.
    """
    denominator = np.sqrt(np.abs(acorr1) * np.abs(acorr2))
    return np.divide(np.abs(corr), denominator, out=np.full(denominator.shape, np.nan), where=denominator > 0)


def phase_degrees(values: np.ndarray) -> np.ndarray:
    """np.rad2deg(np.angle(x))"""
    return np.rad2deg(np.angle(values))


def to_local_times(timestamps: np.ndarray, timezone: ZoneInfo) -> pd.DatetimeIndex:
    times = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(timezone)
    return times.tz_localize(None)


def spectrogram_grid(timestamps: np.ndarray, power: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(column times, columns x channels) of linear power. Gaps stay blank; long spans are averaged."""
    if timestamps.size < 2:
        return timestamps, power

    steps = np.diff(timestamps)
    step = float(np.median(steps))
    if timestamps.size <= MAX_SPECTROGRAM_COLUMNS and steps.max() <= 3 * step:
        return timestamps, power

    # Bins wider than the sample step, so normal timing jitter leaves no empty columns.
    start, end = float(timestamps[0]), float(timestamps[-1]) + step
    columns = max(1, min(MAX_SPECTROGRAM_COLUMNS, int((end - start) / (1.5 * step))))
    positions = np.minimum(((timestamps - start) / (end - start) * columns).astype(int), columns - 1)
    totals = np.zeros((columns, power.shape[1]))
    counts = np.zeros(columns)
    np.add.at(totals, positions, power)
    np.add.at(counts, positions, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        grid = totals / counts[:, None]
    return start + (np.arange(columns) + 0.5) * (end - start) / columns, grid


# ------------------------------------------------------------------------------
# Chart helpers
# ------------------------------------------------------------------------------
def _new_figure(rows: int, height: int, *, row_heights: list[float] | None = None):
    from plotly.subplots import make_subplots

    figure = make_subplots(
        rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.022 if rows > 1 else 0.0, row_heights=row_heights
    )
    figure.update_layout(
        height=height,
        margin={"l": 78, "r": 28, "t": 34, "b": 52},
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font={"color": TEXT_SECONDARY, "size": 12},
        hovermode="x unified",
        hoverlabel={"bgcolor": "#14294a", "font": {"color": TEXT_PRIMARY}, "bordercolor": GRID},
        legend={"orientation": "h", "x": 1, "xanchor": "right", "y": 1.0, "yanchor": "bottom"},
        showlegend=False,
    )
    figure.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, ticks="", showspikes=True, spikemode="across",
                        spikethickness=1, spikecolor=TEXT_SECONDARY, spikedash="solid")
    figure.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, ticks="", exponentformat="SI")
    return figure


def _axis_title(figure, row: int, title: str) -> None:
    figure.update_yaxes(title_text=title, title_font={"color": TEXT_PRIMARY, "size": 12}, row=row, col=1)


def _line(x, y, name: str, color: str = SERIES_1, *, gl: bool = False, width: float = 2.0, legend: bool = False):
    from plotly import graph_objects as go

    trace = go.Scattergl if gl else go.Scatter
    return trace(x=x, y=y, name=name, mode="lines", line={"color": color, "width": width}, showlegend=legend)


def _dots(x, y, name: str, *, gl: bool = False, size: int = 5):
    from plotly import graph_objects as go

    trace = go.Scattergl if gl else go.Scatter
    return trace(x=x, y=y, name=name, mode="markers", marker={"color": SERIES_1, "size": size}, showlegend=False)


def _add_spectrogram(figure, row: int, timestamps, corr, frequencies, timezone: ZoneInfo) -> None:
    from plotly import graph_objects as go

    # Average in linear power, then convert: averaging dB values would hide short bursts.
    column_times, linear = spectrogram_grid(timestamps, np.abs(corr))
    with np.errstate(invalid="ignore"):
        grid = np.where(np.isfinite(linear), power_db(linear), np.nan)
    limits = np.nanpercentile(grid, [2, 98])
    axis_name = "yaxis" if row == 1 else f"yaxis{row}"
    domain = figure.layout[axis_name].domain
    figure.add_trace(
        go.Heatmap(
            x=to_local_times(column_times, timezone),
            y=np.round(frequencies, 3),
            z=np.round(grid.T, 1),
            zmin=float(limits[0]),
            zmax=float(limits[1]),
            colorscale=HEATMAP_SCALE,
            hoverongaps=False,
            hovertemplate="%{x|%H:%M:%S}<br>%{y:.2f} MHz<br>%{z:.1f} dB<extra></extra>",
            colorbar={
                "title": {"text": "dB", "font": {"color": TEXT_PRIMARY}},
                "thickness": 12,
                "outlinewidth": 0,
                "y": sum(domain) / 2,
                "len": domain[1] - domain[0],
                "yanchor": "middle",
            },
        ),
        row=row,
        col=1,
    )
    _axis_title(figure, row, "Frequency (MHz)")


def _finish(figure, row: int, x_title: str, revision: str, time_format: str | None = None, times=None) -> dict:
    figure.update_xaxes(title_text=x_title, title_font={"color": TEXT_PRIMARY, "size": 12}, row=row, col=1)
    if time_format:
        figure.update_xaxes(tickformat=time_format, hoverformat="%H:%M:%S", range=[times[0], times[-1]])
    # Keeps the operator's zoom across page refreshes until new data arrives.
    figure.update_layout(uirevision=revision)
    return figure.to_plotly_json()


# ------------------------------------------------------------------------------
# The four plots
# ------------------------------------------------------------------------------
def spectrum_figure(arrays: dict[str, np.ndarray], index: int, band: Band, revision: str) -> dict:
    """pearson_CC: Pearson_Corr_coeff.png. One spectrum against channel number and frequency."""
    corr = arrays["corr_spec_avg"][index]
    acorr1 = arrays["acorr_spec_avg1"][index]
    acorr2 = arrays["acorr_spec_avg2"][index]
    channels = np.arange(corr.size)

    figure = _new_figure(5, 980)
    figure.add_trace(_line(channels, power_db(acorr1), "Auto-corr. 1"), row=1, col=1)
    _axis_title(figure, 1, "Auto-corr. 1 (dB)")
    figure.add_trace(_line(channels, power_db(acorr2), "Auto-corr. 2"), row=2, col=1)
    _axis_title(figure, 2, "Auto-corr. 2 (dB)")
    figure.add_trace(_line(channels, power_db(corr), "Cross power"), row=3, col=1)
    _axis_title(figure, 3, "Cross power (dB)")
    figure.add_trace(_line(channels, pearson_magnitude(corr, acorr1, acorr2), "Pearson |\u03c1|"), row=4, col=1)
    figure.update_yaxes(rangemode="tozero", row=4, col=1)
    _axis_title(figure, 4, "Pearson |\u03c1|")
    figure.add_trace(_dots(channels, phase_degrees(corr), "Phase"), row=5, col=1)
    figure.update_yaxes(range=[-180, 180], tickvals=[-180, -90, 0, 90, 180], row=5, col=1)
    _axis_title(figure, 5, "Phase (deg)")
    frequencies = band.channel_frequencies(corr.size)
    tick_channels = np.arange(0, corr.size, 25)
    figure.update_xaxes(
        tickvals=tick_channels,
        ticktext=[f"{channel}<br>{frequencies[channel]:.1f}" for channel in tick_channels],
        range=[-1, corr.size],
    )
    return _finish(figure, 5, "Channel number (top) and frequency in MHz (bottom)", revision)


def file_figure(arrays: dict[str, np.ndarray], band: Band, timezone: ZoneInfo, revision: str) -> dict:
    """cc_tseries_phase: cc_tseries_phase.png. One file, averaged over every channel."""
    timestamps = arrays["timestamp"]
    corr = arrays["corr_spec_avg"]
    average = np.mean(corr, axis=1)
    times = to_local_times(timestamps, timezone)

    figure = _new_figure(3, 780, row_heights=[1, 1.25, 1])
    figure.add_trace(_line(times, average.real, "Real", legend=True), row=1, col=1)
    figure.add_trace(_line(times, average.imag, "Imaginary", SERIES_2, legend=True), row=1, col=1)
    figure.update_layout(showlegend=True)
    _axis_title(figure, 1, "Cross-corr. (arb.)")
    _add_spectrogram(figure, 2, timestamps, corr, band.channel_frequencies(corr.shape[1]), timezone)
    figure.add_trace(_line(times, phase_degrees(average), "Phase"), row=3, col=1)
    figure.add_hline(y=0, line={"color": TEXT_SECONDARY, "width": 1, "dash": "dash"}, row=3, col=1)
    _axis_title(figure, 3, "Phase (deg)")
    return _finish(figure, 3, "Time (IST)", revision, "%H:%M:%S", times)


def spectrogram_figure(arrays: dict[str, np.ndarray], band: Band, timezone: ZoneInfo, revision: str) -> dict:
    """spectrogram: spectrogram.png. Time against frequency for one file."""
    corr = arrays["corr_spec_avg"]
    figure = _new_figure(1, 560)
    _add_spectrogram(figure, 1, arrays["timestamp"], corr, band.channel_frequencies(corr.shape[1]), timezone)
    return _finish(figure, 1, "Time (IST)", revision, "%H:%M:%S", to_local_times(arrays["timestamp"], timezone))


def window_figure(
    timestamps: np.ndarray,
    corr: np.ndarray,
    pa_power: np.ndarray,
    band: Band,
    channel_range: tuple[int, int],
    timezone: ZoneInfo,
    revision: str,
) -> dict:
    """one_hr_plot: one_hr_plot.png. The last hour over a slice of the band."""
    channel_count = corr.shape[1]
    start_channel = max(0, min(channel_range[0], channel_count - 1))
    stop_channel = max(start_channel + 1, min(channel_range[1], channel_count))
    corr_slice = corr[:, start_channel:stop_channel]
    average = np.mean(corr_slice, axis=1)
    times = to_local_times(timestamps, timezone)

    figure = _new_figure(5, 1080, row_heights=[1, 1, 1, 1.35, 1])
    figure.add_trace(
        _line(times, np.mean(pa_power[:, start_channel:stop_channel], axis=1), "PA power", gl=True, width=1.2),
        row=1,
        col=1,
    )
    _axis_title(figure, 1, "PA power (arb.)")
    figure.add_trace(_line(times, np.abs(average), "Corr. power", gl=True, width=1.2), row=2, col=1)
    _axis_title(figure, 2, "Corr. power (arb.)")
    figure.add_trace(_line(times, average.real, "Real", gl=True, width=1.2, legend=True), row=3, col=1)
    figure.add_trace(_line(times, average.imag, "Imaginary", SERIES_2, gl=True, width=1.2, legend=True), row=3, col=1)
    figure.update_layout(showlegend=True)
    _axis_title(figure, 3, "Cross-corr. (arb.)")
    _add_spectrogram(
        figure, 4, timestamps, corr_slice, band.channel_frequencies(channel_count)[start_channel:stop_channel], timezone
    )
    figure.add_trace(_dots(times, phase_degrees(average), "Phase", gl=True, size=3), row=5, col=1)
    figure.add_hline(y=0, line={"color": TEXT_SECONDARY, "width": 1, "dash": "dash"}, row=5, col=1)
    figure.update_yaxes(range=[-180, 180], tickvals=[-180, -90, 0, 90, 180], row=5, col=1)
    _axis_title(figure, 5, "Phase (deg)")
    return _finish(figure, 5, "Time (IST)", revision, "%H:%M", times)
