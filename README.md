# Repo of deep learning application in coherent wireless receivers

## Requirements
* Pytorch (with Numpy, Scipy etc.)
* Matlab (only for .m file)
* Google Colaboratory (optional)
## Symbol timing recovery
`sig_gen5_new_timing.m` is used for data generation

`CNN_for_symbol_timing.ipynb` uses a 2D CNN as a classifier to estimate the best timing offset. It works with passband signal, so no complex value is involved. May run into few errors due to some dependency lost. Will be fix later (I hope).

## Carrier frequency recovery
`CFO_recovery_training.ipynb` trains the CNN with synthesized data. The frequency recovery module is basicly a PLL, but the phase error detector is replaced with the CNN. One advantage of using synthesized data is that in every epoch of training, a set of new traing data can be generated, so the overfitting is not an issue here.

`CFO_recovery_for_sc80.ipynb` tests the pretrained CNN with sea trial data.

**Note** that it works in basedband, but only the signal phase components are considered, and the signal amplitudes are ignored. This is only resonable when a perfect timing has been done and no visable ISI in the received signal. 

## Equalizer
`Passband_Demodulation.ipynb` is in progress. QPSK demodulation in passband. No complex-valued signal is involved.
Currently can only demodulate signal when carrier frequency is integer multiple of the symbol rate.
To remove this limitation, a phase compensation is required.

`equalization.ipynb` is obsolete. The traing signal is synthesized using Numpy only, so no Matlab is required. 
Currently a fractionally spaced, decision feedback equalizer has been built for a multipath transmission, however the performance is not as good as expected.

## To do
* Demodulation in passband with phase compensation.
* Integrate CFO compensation.
* Integrate DFE for multipath


