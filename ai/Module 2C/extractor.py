"""Numerical STFT, log-Mel and LFCC representations for Module 2B."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from spectral_config import (
    EPSILON,
    HOP_LENGTH,
    N_FFT,
    N_LFCC,
    N_MELS,
    SAMPLE_RATE,
    WIN_LENGTH,
)


class SpectralExtractor:
    """
    Convert one waveform into a numerical spectral representation.

    Output shape:
        [frequency_bins, time_frames]

    Supported representations:
        - stft
        - logmel
        - lfcc
    """

    VALID_TYPES = {"stft", "logmel", "lfcc"}

    def __init__(
        self,
        spectral_type: str = "logmel",
        sample_rate: int = SAMPLE_RATE,
        n_fft: int = N_FFT,
        win_length: int = WIN_LENGTH,
        hop_length: int = HOP_LENGTH,
        n_mels: int = N_MELS,
        n_lfcc: int = N_LFCC,
    ) -> None:
        if spectral_type not in self.VALID_TYPES:
            raise ValueError(
                f"spectral_type must be one of {sorted(self.VALID_TYPES)}"
            )

        self.spectral_type = spectral_type
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_lfcc = n_lfcc

        self.window = torch.hann_window(win_length)

    def _stft_power(self, waveform: Tensor) -> Tensor:
        """Return numerically stable STFT power spectrum [F, T]."""
        audio = torch.as_tensor(waveform, dtype=torch.float32).flatten()

        if audio.numel() == 0:
            raise ValueError("waveform is empty")

        if not torch.isfinite(audio).all():
            raise ValueError("waveform contains NaN or Inf")

        if audio.numel() < self.win_length:
            audio = torch.nn.functional.pad(
                audio,
                (0, self.win_length - audio.numel()),
            )

        window = self.window.to(device=audio.device, dtype=audio.dtype)

        spectrum = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=True,
            return_complex=True,
        )

        power = spectrum.abs().square()

        return power.clamp_min(EPSILON)

    def _mel_filterbank(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """Create a Slaney-style triangular Mel filterbank."""

        def hz_to_mel(hz: Tensor) -> Tensor:
            return 2595.0 * torch.log10(1.0 + hz / 700.0)

        def mel_to_hz(mel: Tensor) -> Tensor:
            return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

        low_hz = torch.tensor(
            0.0,
            device=device,
            dtype=dtype,
        )

        high_hz = torch.tensor(
            self.sample_rate / 2.0,
            device=device,
            dtype=dtype,
        )

        mel_points = torch.linspace(
            hz_to_mel(low_hz),
            hz_to_mel(high_hz),
            self.n_mels + 2,
            device=device,
            dtype=dtype,
        )

        hz_points = mel_to_hz(mel_points)

        fft_bins = torch.floor(
            (self.n_fft + 1) * hz_points / self.sample_rate
        ).long()

        filterbank = torch.zeros(
            self.n_mels,
            self.n_fft // 2 + 1,
            device=device,
            dtype=dtype,
        )

        for i in range(self.n_mels):
            left = int(fft_bins[i].item())
            center = int(fft_bins[i + 1].item())
            right = int(fft_bins[i + 2].item())

            center = max(center, left + 1)
            right = max(right, center + 1)

            center = min(center, self.n_fft // 2 + 1)
            right = min(right, self.n_fft // 2 + 1)

            if center > left:
                rising = torch.arange(
                    left,
                    center,
                    device=device,
                    dtype=dtype,
                )

                filterbank[i, left:center] = (
                    rising - left
                ) / max(center - left, 1)

            if right > center:
                falling = torch.arange(
                    center,
                    right,
                    device=device,
                    dtype=dtype,
                )

                filterbank[i, center:right] = (
                    right - falling
                ) / max(right - center, 1)

        return filterbank

    def _linear_filterbank(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """
        Create a linear-frequency triangular filterbank for LFCC.

        Unlike Mel features, the filters are uniformly spaced in Hz.
        """

        fft_bins = self.n_fft // 2 + 1

        frequencies = torch.linspace(
            0.0,
            self.sample_rate / 2.0,
            self.n_lfcc + 2,
            device=device,
            dtype=dtype,
        )

        fft_frequencies = torch.linspace(
            0.0,
            self.sample_rate / 2.0,
            fft_bins,
            device=device,
            dtype=dtype,
        )

        filterbank = torch.zeros(
            self.n_lfcc,
            fft_bins,
            device=device,
            dtype=dtype,
        )

        for i in range(self.n_lfcc):
            left = frequencies[i]
            center = frequencies[i + 1]
            right = frequencies[i + 2]

            rising_denominator = (center - left).clamp_min(EPSILON)
            falling_denominator = (right - center).clamp_min(EPSILON)

            rising = (fft_frequencies - left) / rising_denominator
            falling = (right - fft_frequencies) / falling_denominator

            filterbank[i] = torch.clamp(
                torch.minimum(rising, falling),
                min=0.0,
            )

        return filterbank

    def _dct(self, values: Tensor) -> Tensor:
        """Apply an orthogonal DCT-II across the filter dimension."""
        n = values.shape[0]

        k = torch.arange(
            n,
            device=values.device,
            dtype=values.dtype,
        ).unsqueeze(1)

        n_index = torch.arange(
            n,
            device=values.device,
            dtype=values.dtype,
        ).unsqueeze(0)

        basis = torch.cos(
            math.pi
            / n
            * (n_index + 0.5)
            * k
        )

        basis[0] *= 1.0 / math.sqrt(n)

        if n > 1:
            basis[1:] *= math.sqrt(2.0 / n)

        return basis @ values

    def __call__(
        self,
        waveform: Tensor | list[float],
    ) -> Tensor:
        """Extract one [F, T] spectral representation."""

        power = self._stft_power(torch.as_tensor(waveform))

        if self.spectral_type == "stft":
            # Explicitly log-power STFT.
            features = torch.log(power.clamp_min(EPSILON))

        elif self.spectral_type == "logmel":
            filterbank = self._mel_filterbank(
                power.device,
                power.dtype,
            )

            mel_energy = filterbank @ power

            features = torch.log(
                mel_energy.clamp_min(EPSILON)
            )

        else:
            # LFCC:
            # power spectrum
            # -> linear-frequency filterbank
            # -> log filterbank energies
            # -> DCT-II
            filterbank = self._linear_filterbank(
                power.device,
                power.dtype,
            )

            filterbank_energy = filterbank @ power

            log_filterbank = torch.log(
                filterbank_energy.clamp_min(EPSILON)
            )

            features = self._dct(log_filterbank)

        features = features.float()

        if features.ndim != 2:
            raise RuntimeError(
                f"expected [F, T], got {tuple(features.shape)}"
            )

        if not torch.isfinite(features).all():
            raise ValueError(
                "spectral representation contains NaN or Inf"
            )

        return features

    def batch(self, waveforms: Tensor) -> Tensor:
        """Extract a batch of waveforms."""
        return torch.stack(
            [self(waveform) for waveform in waveforms]
        )