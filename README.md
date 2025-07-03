# BinauralRendering
This script provides a complete pipeline for rendering binaural audio from Ambisonic recordings using the sound_field_analysis library. It processes DRIR and HRIR files in the Spherical Harmonic domain, applies optional filters, and generates .wav files for head orientations.

# How to Run
'''
conda env create -f environment.yml
'''

'''
conda activate sfa
'''

'''
python binaural.py
'''

# A Note on Warnings
You may see a RuntimeWarning: invalid value encountered in divide, which appears when the script applies the Spherical Head Filter (originating from the apply_shf function's helpers). This is caused by an unavoidable division-by-zero in the filter's physics model, where the derivative of a spherical Hankel function (dsphankel2) used in a denominator becomes zero at specific frequencies. The library's author anticipated this and included a fix, making the warning harmless to the final output.
