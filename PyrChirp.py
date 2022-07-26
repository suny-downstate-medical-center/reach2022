from geom import *
# from neuron import h 
# h.load_file('stdrun.hoc')
# cell = PVC(0,0,0,0)
# h.psection()
cell = PyrAdr(0,0,0,0)
from chirpUtils import getChirp
from pylab import fft
soma_seg = cell.soma(0.5)
seg = cell.soma(0.5)
stim = h.IClamp(seg)
amp = 0.01 
t0 = 20
delay = 5
Fs = 1000
sampr = 40e3 
f0 = 0.5 #args.f0
f1 = 20 #args.f1
soma_v = h.Vector().record(soma_seg._ref_v) 
seg_v = h.Vector().record(seg._ref_v) 
time = h.Vector().record(h._ref_t)
I, t = getChirp(f0, f1, t0, amp, Fs, delay, offset=0.0)
i = h.Vector().record(h.IClamp[1]._ref_i)
stim.amp = 0
stim.dur = (t0+delay*2) * Fs + 1
I.play(stim._ref_amp, t)
## run simulation
h.celsius = 34
h.tstop = (t0+delay*2) * Fs + 1
# print('running ' + args.cellModel + ' ' + args.section + ' f0-' + str(round(args.f0)) + ' f1-' + str(round(args.f1)))
h.run()
v_trim = [v for v, T in zip(seg_v, time) if int((delay)*1000) < T < int((delay+t0)*1000)] 
i_trim = [x for x, T in zip(i,time) if int((delay)*1000) < T < int((delay+t0)*1000)] 
time_trim = [T for v, T in zip(soma_v, time) if int((delay)*1000) < T < int((delay+t0)*1000)] 
current = i_trim
v = v_trim 
#current = current[int(delay*sampr - 0.5*sampr+1):-int(delay*sampr- 0.5*sampr)] 
current = np.hstack((np.repeat(current[0],int(delay*sampr)),current, np.repeat(current[-1], int(delay*sampr)))) 
current = current - np.mean(current) 
#v = v[int(delay*sampr - 0.5*sampr)+1:-int(delay*sampr - 0.5*sampr)] 
v = v - np.mean(v) 
v = np.hstack((np.repeat(0,int(delay*sampr)), v, np.repeat(0, int(delay*sampr)))) 
f_current = (fft(current)/len(current))[0:int(len(current)/2)] 
f_cis = (fft(v)/len(v))[0:int(len(v)/2)] 
z = f_cis / f_current 
phase = np.arctan2(np.imag(z), np.real(z))
Freq       = np.linspace(0.0, sampr/2.0, len(z))
zRes       = np.real(z)
zReact     = np.imag(z)
zamp = abs(z)
mask = (Freq >= 0.5) & (Freq <= f1)
zResAmp    = np.max(zamp)
zResFreq   = Freq[np.argmax(zamp)]
Qfactor    = zResAmp / zamp[0]
fVar       = np.std(zamp) / np.mean(zamp)
peak_to_peak = np.max(v) - np.min(v)
## smoothing
# bwinsz = 10
# fblur = np.array([1.0/bwinsz for i in range(bwinsz)])
# zamp = convolve(zamp,fblur,'same')
# phase = convolve(phase, fblur, 'same')
Freq, zamp, phase, zRes, zReact, z = Freq[mask], zamp[mask], phase[mask], zRes[mask], zReact[mask], z[mask]

from matplotlib import pyplot as plt
plt.ion()
plt.plot(Freq, zamp)