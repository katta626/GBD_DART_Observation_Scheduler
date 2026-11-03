"""Diagnostic plots for the DIAGNOSTIC PLOTS tab, read directly from the PDR HDF5 files.

Reproduces the four plots of the original scripts in live_PDR_plot.zip (pearson_CC,
cc_tseries_phase, spectrogram, one_hr_plot) with corrected labels. Each plot function
returns PNG bytes, so the app can cache a finished image and only redraw it when a new
file arrives; that keeps the tab fast while the page refreshes every few seconds.
"""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

# The acquisition leaves 96-byte HDF5 files (a header with no data) next to the real ones.
MIN_VALID_FILE_BYTES = 1024
MAX_SPECTROGRAM_COLUMNS = 720
SECONDS_PER_DAY = 86400.0
PLOT_DPI = 120

SPECTRUM_DATASETS = ("timestamp", "corr_spec_avg", "acorr_spec_avg1", "acorr_spec_avg2")
WINDOW_DATASETS = ("timestamp", "corr_spec_avg", "p_sum_avg")

# Files that are empty, truncated, or still being written.
UNREADABLE_FILE_ERRORS = (OSError, KeyError, ValueError)


@dataclass(frozen=True)
class Band:
    center_mhz: float = 200.0
    bandwidth_mhz: float = 16.5
    channel0_is_high: bool = False

    def channel_frequencies(self, channel_count: int) -> np.ndarray:
        """Centre frequency (MHz) of every channel, in channel order."""
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
    """Per-channel rotation from rf_path_phase.dat (degrees), as in the original script."""
    phase_degrees = np.loadtxt(calibration_path, dtype=float, delimiter=",")
    if phase_degrees.size != channel_count:
        raise ValueError(f"expected {channel_count} channels, found {phase_degrees.size}")
    return np.exp(1j * np.deg2rad(phase_degrees))


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
    """10*log10(|x|^2), as in the original scripts, without log(0) warnings."""
    return 10 * np.log10(np.maximum(np.abs(values) ** 2, np.finfo(float).tiny))


def coherence(corr: np.ndarray, acorr1: np.ndarray, acorr2: np.ndarray) -> np.ndarray:
    """Normalised correlation |rho| = |cross| / sqrt(|auto1| * |auto2|), from 0 to 1.

    The original plotted the complex ratio, and matplotlib silently kept the real part,
    which swings with the phase instead of showing how correlated the two signals are.
    """
    denominator = np.sqrt(np.abs(acorr1) * np.abs(acorr2))
    return np.divide(np.abs(corr), denominator, out=np.full(denominator.shape, np.nan), where=denominator > 0)


def phase_degrees(values: np.ndarray) -> np.ndarray:
    return np.rad2deg(np.angle(values))


def spectrogram_grid(timestamps: np.ndarray, power: np.ndarray) -> tuple[np.ndarray, float, float]:
    """(columns x channels, start, end): real gaps stay blank, long spans are averaged down.

    A browser or image cannot show more columns than it has pixels; one hour is ~7000 spectra.
    """
    if timestamps.size == 1:
        return power, float(timestamps[0]), float(timestamps[0]) + 1.0

    steps = np.diff(timestamps)
    step = float(np.median(steps))
    start, end = float(timestamps[0]), float(timestamps[-1]) + step
    if timestamps.size <= MAX_SPECTROGRAM_COLUMNS and steps.max() <= 3 * step:
        return power, start, end

    # Bins wider than the sample step, so normal timing jitter does not leave empty columns.
    columns = max(1, min(MAX_SPECTROGRAM_COLUMNS, int((end - start) / (1.5 * step))))
    positions = np.minimum(((timestamps - start) / (end - start) * columns).astype(int), columns - 1)
    totals = np.zeros((columns, power.shape[1]))
    counts = np.zeros(columns)
    np.add.at(totals, positions, power)
    np.add.at(counts, positions, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return totals / counts[:, None], start, end


# ------------------------------------------------------------------------------
# Plot helpers
# ------------------------------------------------------------------------------
def _new_figure(rows: int, height: float, *, height_ratios: list[float] | None = None):
    from matplotlib.figure import Figure

    figure = Figure(figsize=(11, height), layout="constrained")
    axes = figure.subplots(rows, 1, sharex=True, squeeze=False, height_ratios=height_ratios)[:, 0]
    for axis in axes:
        axis.grid(True, alpha=0.35)
    return figure, axes


def _to_png(figure) -> bytes:
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=PLOT_DPI)
    return buffer.getvalue()


def _format_time_axis(axis, timezone: ZoneInfo, time_format: str) -> None:
    from matplotlib import dates

    axis.xaxis_date(tz=timezone)
    axis.xaxis.set_major_locator(dates.AutoDateLocator(tz=timezone, minticks=5, maxticks=10))
    axis.xaxis.set_major_formatter(dates.DateFormatter(time_format, tz=timezone))
    axis.set_xlabel(f"Time ({timezone.key.split('/')[-1] if timezone.key != 'Asia/Kolkata' else 'IST'})")


def _plot_phase_over_time(axis, days: np.ndarray, phase: np.ndarray, fixed_range: bool) -> None:
    # Dots, not a line: phase wraps at +/-180 and a line would draw false vertical jumps.
    axis.plot(days, phase, ".", markersize=3)
    axis.axhline(0, color="red", linewidth=1)
    axis.set_ylabel("Phase (deg)")
    if fixed_range:
        axis.set_ylim(-180, 180)
        axis.set_yticks([-180, -90, 0, 90, 180])


def _plot_spectrogram(axis, timestamps: np.ndarray, power: np.ndarray, frequencies: np.ndarray) -> None:
    grid, start, end = spectrogram_grid(timestamps, power)
    # Sort channels so the image runs from low to high frequency whichever way the band runs.
    grid = grid[:, np.argsort(frequencies)]
    half_width = abs(frequencies[1] - frequencies[0]) / 2 if frequencies.size > 1 else 0.5
    limits = np.nanpercentile(grid, [2, 98]) if np.any(np.isfinite(grid)) else (None, None)
    image = axis.imshow(
        grid.T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=(
            start / SECONDS_PER_DAY,
            end / SECONDS_PER_DAY,
            frequencies.min() - half_width,
            frequencies.max() + half_width,
        ),
        vmin=limits[0],
        vmax=limits[1],
    )
    axis.grid(False)
    axis.set_ylabel("Frequency (MHz)")
    colorbar_axis = axis.inset_axes([1.012, 0.0, 0.015, 1.0])
    axis.figure.colorbar(image, cax=colorbar_axis, label="Cross power (dB, arb.)")


# ------------------------------------------------------------------------------
# The four plots
# ------------------------------------------------------------------------------
def plot_spectrum(arrays: dict[str, np.ndarray], index: int, band: Band, timezone: ZoneInfo, file_name: str) -> bytes:
    """Original Pearson_Corr_coeff.png: one spectrum across the band."""
    from datetime import datetime

    corr = arrays["corr_spec_avg"][index]
    acorr1 = arrays["acorr_spec_avg1"][index]
    acorr2 = arrays["acorr_spec_avg2"][index]
    frequencies = band.channel_frequencies(corr.size)
    spectrum_time = datetime.fromtimestamp(float(arrays["timestamp"][index]), timezone)

    figure, axes = _new_figure(5, 10.5)
    figure.suptitle(
        f"Spectrum {index + 1} of {arrays['corr_spec_avg'].shape[0]} · "
        f"{spectrum_time:%Y-%m-%d %H:%M:%S} IST\n{file_name}",
        fontsize=11,
    )
    axes[0].plot(frequencies, power_db(acorr1))
    axes[0].set_ylabel("Auto-corr. 1\npower (dB, arb.)")
    axes[1].plot(frequencies, power_db(acorr2))
    axes[1].set_ylabel("Auto-corr. 2\npower (dB, arb.)")
    axes[2].plot(frequencies, power_db(corr))
    axes[2].set_ylabel("Cross power\n(dB, arb.)")
    rho = coherence(corr, acorr1, acorr2)
    axes[3].plot(frequencies, rho)
    # Scale to the data: |rho| is often ~0.02 on sky, and a fixed 0-1 axis would flatten it.
    peak = np.nanmax(rho) if np.any(np.isfinite(rho)) else 1.0
    axes[3].set_ylim(0, min(1.0, max(0.05, peak * 1.15)))
    axes[3].set_ylabel("Pearson correlation\n|ρ|")
    axes[4].plot(frequencies, phase_degrees(corr), ".", markersize=4)
    axes[4].set_ylim(-180, 180)
    axes[4].set_yticks([-180, -90, 0, 90, 180])
    axes[4].set_ylabel("Cross phase (deg)")
    axes[4].set_xlabel("Frequency (MHz)")
    axes[4].set_xlim(frequencies.min(), frequencies.max())

    # Channel numbers along the top, for comparing with the acquisition logs.
    first, step = frequencies[0], frequencies[1] - frequencies[0]
    channel_axis = axes[0].secondary_xaxis(
        "top", functions=(lambda mhz: (mhz - first) / step, lambda channel: first + channel * step)
    )
    channel_axis.set_xlabel("Channel number")
    return _to_png(figure)


def plot_file_summary(arrays: dict[str, np.ndarray], band: Band, timezone: ZoneInfo, file_name: str) -> bytes:
    """Original cc_tseries_phase.png: one file, averaged over the band."""
    timestamps = arrays["timestamp"]
    corr = arrays["corr_spec_avg"]
    average = np.mean(corr, axis=1)
    days = timestamps / SECONDS_PER_DAY

    figure, axes = _new_figure(3, 8.5, height_ratios=[1, 1.2, 1])
    figure.suptitle(f"Band-averaged cross-correlation · {file_name}", fontsize=11)
    axes[0].plot(days, average.real, label="Real")
    axes[0].plot(days, average.imag, label="Imaginary")
    axes[0].set_ylabel("Cross-corr. (arb.)")
    axes[0].legend(loc="center right", ncols=2)
    _plot_spectrogram(axes[1], timestamps, power_db(corr), band.channel_frequencies(corr.shape[1]))
    _plot_phase_over_time(axes[2], days, phase_degrees(average), fixed_range=False)
    _format_time_axis(axes[2], timezone, "%H:%M:%S")
    return _to_png(figure)


def plot_spectrogram(arrays: dict[str, np.ndarray], band: Band, timezone: ZoneInfo, file_name: str) -> bytes:
    """Original spectrogram.png: time against frequency for one file."""
    corr = arrays["corr_spec_avg"]
    figure, axes = _new_figure(1, 5.5)
    figure.suptitle(f"Cross-correlation spectrogram · {file_name}", fontsize=11)
    _plot_spectrogram(axes[0], arrays["timestamp"], power_db(corr), band.channel_frequencies(corr.shape[1]))
    _format_time_axis(axes[0], timezone, "%H:%M:%S")
    return _to_png(figure)


def plot_window(
    timestamps: np.ndarray,
    corr: np.ndarray,
    pa_power: np.ndarray,
    band: Band,
    channel_range: tuple[int, int],
    timezone: ZoneInfo,
    *,
    calibration_note: str,
) -> bytes:
    """Original one_hr_plot.png: the last hour over a slice of the band."""
    from datetime import datetime

    channel_count = corr.shape[1]
    start_channel = max(0, min(channel_range[0], channel_count - 1))
    stop_channel = max(start_channel + 1, min(channel_range[1], channel_count))
    frequencies = band.channel_frequencies(channel_count)[start_channel:stop_channel]
    corr_slice = corr[:, start_channel:stop_channel]
    average = np.mean(corr_slice, axis=1)
    days = timestamps / SECONDS_PER_DAY
    first = datetime.fromtimestamp(float(timestamps[0]), timezone)
    last = datetime.fromtimestamp(float(timestamps[-1]), timezone)

    figure, axes = _new_figure(5, 11.5, height_ratios=[1, 1, 1, 1.4, 1])
    figure.suptitle(
        f"{first:%Y-%m-%d %H:%M:%S} to {last:%H:%M:%S} IST · channels {start_channel}:{stop_channel} "
        f"({frequencies.min():.2f}–{frequencies.max():.2f} MHz) · {calibration_note}",
        fontsize=11,
    )
    axes[0].plot(days, np.mean(pa_power[:, start_channel:stop_channel], axis=1), linewidth=0.8)
    axes[0].set_ylabel("PA power (arb.)")
    axes[1].plot(days, power_db(average), linewidth=0.8)
    axes[1].set_ylabel("Cross power\n(dB, arb.)")
    axes[2].plot(days, average.real, linewidth=0.8, label="Real")
    axes[2].plot(days, average.imag, linewidth=0.8, label="Imaginary")
    axes[2].set_ylabel("Cross-corr. (arb.)")
    axes[2].legend(loc="upper right")
    _plot_spectrogram(axes[3], timestamps, power_db(corr_slice), frequencies)
    _plot_phase_over_time(axes[4], days, phase_degrees(average), fixed_range=True)
    _format_time_axis(axes[4], timezone, "%H:%M")
    return _to_png(figure)
