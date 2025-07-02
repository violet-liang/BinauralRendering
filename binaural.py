import numpy as np
import h5py
import sys
sys.path.insert(0, "../")
from sound_field_analysis import io, gen, process, sph, utils
from scipy.io import wavfile

# ----------------------
# Configuration Section
# ----------------------
sh_max_order    = 8      # Max spherical harmonic rendering order
is_real_sh      = False  # Use real or complex spherical harmonics
rf_nfft         = 2048   # Radial filter length (samples)
rf_amp_max_db   = 20     # Radial filter soft limiting (dB)
is_apply_rfi    = True   # Apply radial filter improvement
is_apply_sht    = True   # Apply Spherical Harmonic Tapering
is_apply_shf    = True   # Apply Spherical Head Filter
shf_nfft        = 256    # Spherical Head Filter length (samples)
#azim_offset_deg = 0   # azimuth head rotation offset in degrees (to make the listener face the sound source)
pre_azims_deg   = [0,45,90,135,180,225,270,315]    # Preview head orientations (deg)
pre_len_s       = 2      # Preview auralization length (s)
pre_src_file    = "data/audio.wav"  # Preview audio file
sh_kind = "real" if is_real_sh else "complex"


def get_azim_offset(dr_path):
    """calculate azimuth offset so that the head is facing the source."""
    with h5py.File(dr_path, "r") as f:
        source_pos = f["SourcePosition"][:]
        source_x = source_pos[0,0]
        source_y = source_pos[0,1]
        azimuth_rad = np.arctan2(source_y, source_x)
        azimuth_deg = np.rad2deg(azimuth_rad)
        azim_offset_deg = -int(azimuth_deg)
    return azim_offset_deg



def processing_length(DRIR, HRIR, rf_nfft, shf_nfft, is_apply_shf):
    """Calculate target processing length for FFT."""
    NFFT = HRIR.l.signal.shape[-1]
    NFFT += DRIR.signal.signal.shape[-1] + rf_nfft
    if is_apply_shf:
        NFFT += shf_nfft
    return NFFT


def sh_coeffs(HRIR, DRIR, NFFT, sh_max_order, sh_kind):
    """Compute SH coefficients for HRIR and DRIR."""
    Hnm = np.stack([
        process.spatFT(
            process.FFT(HRIR.l.signal, fs=int(HRIR.l.fs), NFFT=NFFT, calculate_freqs=False),
            position_grid=HRIR.grid,
            order_max=sh_max_order,
            kind=sh_kind,
        ),
        process.spatFT(
            process.FFT(HRIR.r.signal, fs=int(HRIR.l.fs), NFFT=NFFT, calculate_freqs=False),
            position_grid=HRIR.grid,
            order_max=sh_max_order,
            kind=sh_kind,
        ),
    ])
    Pnm = process.spatFT(
        process.FFT(DRIR.signal.signal, fs=int(HRIR.l.fs), NFFT=NFFT, calculate_freqs=False),
        position_grid=DRIR.grid,
        order_max=sh_max_order,
        kind=sh_kind,
    )
    return Hnm, Pnm


def radial_filters(DRIR, sh_max_order, rf_nfft, FS, rf_amp_max_db, is_apply_rfi):
    """Compute and optionally improve radial filters."""
    dn = gen.radial_filter_fullspec(
        max_order=sh_max_order,
        NFFT=rf_nfft,
        fs=FS,
        array_configuration=DRIR.configuration,
        amp_maxdB=rf_amp_max_db,
    )
    if is_apply_rfi:
        dn, _, dn_delay_samples = process.rfi(dn, kernelSize=rf_nfft)
    else:
        dn_delay_samples = rf_nfft / 2
        dn *= gen.delay_fd(target_length_fd=dn.shape[-1], delay_samples=dn_delay_samples)
    return dn, dn_delay_samples


def apply_sht(dn, sh_max_order):
    """Apply Spherical Harmonic Tapering window to radial filters."""
    dn_sht = gen.tapering_window(max_order=sh_max_order)
    dn_sht = np.repeat(
        dn_sht[:, np.newaxis], np.arange(sh_max_order * 2 + 1, step=2) + 1, axis=0
    )
    return dn * dn_sht


def apply_shf(dn, sh_max_order, NFFT, DRIR, shf_nfft, FS, is_apply_sht):
    """Apply Spherical Head Filter to radial filters."""
    dn_shf = gen.spherical_head_filter_spec(
        max_order=sh_max_order,
        NFFT=shf_nfft,
        fs=FS,
        radius=DRIR.configuration.array_radius,
        is_tapering=is_apply_sht,
    )
    dn_shf_delay_samples = shf_nfft / 2
    dn_shf *= gen.delay_fd(
        target_length_fd=dn_shf.shape[-1], delay_samples=dn_shf_delay_samples
    )
    dn_shf = utils.zero_pad_fd(dn_shf, target_length_td=NFFT)
    dn = dn * dn_shf
    return dn, dn_shf_delay_samples


def compute_brir(Hnm, Pnm, dn, sh_max_order, is_real_sh, azim_offset_deg):
    """Compute Binaural Room Impulse Responses (BRIR) for all head orientations."""
    m = sph.mnArrays(sh_max_order)[0]
    m_rev_id = sph.reverseMnIds(sh_max_order)
    azims_SSR_rad = np.deg2rad(np.arange(0, 360) - azim_offset_deg)  # Rotate
    if is_real_sh:
        Pnm_dn = Pnm * dn
        S = np.zeros([len(azims_SSR_rad), Hnm.shape[0], Hnm.shape[-1]], dtype=Hnm.dtype)
        for azim_id, alpha in enumerate(azims_SSR_rad):
            alpha_cos = np.cos(m * alpha)[:, np.newaxis]
            alpha_sin = np.sin(m * alpha)[:, np.newaxis]
            S[azim_id] = np.sum((alpha_cos * Pnm_dn - alpha_sin * Pnm_dn[m_rev_id]) * Hnm, axis=1)
    else:
        Pnm_dn_Hnm = np.float_power(-1.0, m)[:, np.newaxis] * Pnm[m_rev_id] * dn * Hnm
        S = np.zeros([len(azims_SSR_rad), Hnm.shape[0], Hnm.shape[-1]], dtype=Hnm.dtype)
        for azim_id, alpha in enumerate(azims_SSR_rad):
            alpha_exp = np.exp(-1j * m * alpha)[:, np.newaxis]
            S[azim_id] = np.sum(Pnm_dn_Hnm * alpha_exp, axis=1)
    BRIR = process.iFFT(S)
    BRIR *= 0.9 / np.max(np.abs(BRIR))  # Normalize
    return BRIR


def save_binaural_audio(BRIR, pre_azims_deg, source, FS):
    """Convolve source with BRIR and save binaural audio for each orientation."""
    for azim in pre_azims_deg:
        print(f"Head orientation {azim}°")
        audio_data = process.convolve(source, BRIR[azim])
        if audio_data.shape[0] < audio_data.shape[1]:
            audio_data = audio_data.T
        audio_data = np.nan_to_num(audio_data)
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val
        audio_int16 = (audio_data * 32767).astype(np.int16)
        filename = f"binaural_{azim}deg.wav"
        wavfile.write(filename, int(FS), audio_int16)
        print(f"Saved: {filename}")


def main():

    DRIR = io.read_SOFA_file("data/DRIR_CR1_VSA_1202RS_R.sofa") #sound field
    azim_offset_deg = get_azim_offset("data/DRIR_CR1_VSA_1202RS_R.sofa")
    HRIR = io.read_SOFA_file("data/HRIR_L2702.sofa") 
    FS = int(HRIR.l.fs)  
    
    # Spherical harmonic decomposition
    NFFT = processing_length(DRIR, HRIR, rf_nfft, shf_nfft, is_apply_shf)  
    Hnm, Pnm = sh_coeffs(HRIR, DRIR, NFFT, sh_max_order, sh_kind) 

    # Radial filters
    dn, dn_delay_samples = radial_filters(DRIR, sh_max_order, rf_nfft, FS, rf_amp_max_db, is_apply_rfi)  
    dn = utils.zero_pad_fd(dn, target_length_td=NFFT)  
    dn = np.repeat(dn, np.arange(sh_max_order * 2 + 1, step=2) + 1, axis=0)  # Repeat for SH grades
    if is_apply_sht:
        dn = apply_sht(dn, sh_max_order)  
    dn_shf_delay_samples = 0
    if is_apply_shf:
        dn, dn_shf_delay_samples = apply_shf(dn, sh_max_order, NFFT, DRIR, shf_nfft, FS, is_apply_sht)  
    dn_delay_samples += dn_shf_delay_samples 

    # Binauralize
    BRIR = compute_brir(Hnm, Pnm, dn, sh_max_order, is_real_sh, azim_offset_deg)  
    source, source_fs = io.read_wavefile(pre_src_file)  # Load source audio
    if len(source.shape) > 1:
        source = source[0]  # Use first channel if multi-channel
    source = np.atleast_2d(source[: int(pre_len_s * source_fs)])  # Truncate to preview length
    source = utils.simple_resample(source, original_fs=source_fs, target_fs=FS)  # Resample to match BRIR
    save_binaural_audio(BRIR, pre_azims_deg, source, FS)  # Save binaural audio for each orientation


if __name__ == "__main__":
    main()



