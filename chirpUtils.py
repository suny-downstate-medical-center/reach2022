from neuron import h, init
# h.load_file("/usr/local/nrn//share/nrn/lib/hoc/stdrun.hoc")
h.load_file('stdrun.hoc')
import numpy as np 
from scipy.signal import chirp
from pylab import fft, convolve
from scipy.io import savemat
import json
from scipy.signal import find_peaks
import multiprocessing
import pickle 

# get chirp stim: based on sam's code form evoizhi/sim.py
def getChirp(f0, f1, t0, amp, Fs, delay, offset=0):
    time = np.linspace(0,t0+delay*2, (t0+delay*2)*Fs*40+1)
    chirp_time = np.linspace(0, t0, (t0)*Fs*40+1)
    ch = chirp(chirp_time, f0, t0, f1, method='linear',phi=-90)
    ch = np.hstack((np.zeros(Fs*40*delay), ch, np.zeros(Fs*40*delay)))
    vch = h.Vector(); vch.from_python(ch); vch.mul(amp); vch.add(offset)
    vtt = h.Vector(); vtt.from_python(time); vtt.mul(Fs)
    return vch, vtt

def getRampChirp(f0, f1, t0, amp, Fs, delay, offset=0, slope=None):
    time = np.linspace(0,t0+delay*2, (t0+delay*2)*Fs*40+1)
    chirp_time = np.linspace(0, t0, (t0)*Fs*40+1)
    ch = chirp(chirp_time, f0, t0, f1, method='linear',phi=-90)
    if not slope:
        slope = 1 / t0 
    ramp = np.array([t*slope for t in chirp_time])
    chramp = ch * ramp 
    chramp = np.hstack((np.zeros(Fs*40*delay), chramp, np.zeros(Fs*40*delay)))
    vch = h.Vector(); vch.from_python(chramp); vch.mul(amp); vch.add(offset)
    vtt = h.Vector(); vtt.from_python(time); vtt.mul(Fs)
    return vch, vtt

def getChirpHighFs(f0, f1, t0, amp, Fs, delay):
    time = np.linspace(0,t0+delay*2, (t0+delay*2)*Fs*500+1)
    chirp_time = np.linspace(0, t0, (t0)*Fs*500+1)
    ch = chirp(chirp_time, f0, t0, f1, method='linear',phi=-90)
    ch = np.hstack((np.zeros(Fs*500*delay), ch, np.zeros(Fs*500*delay)))
    vch = h.Vector(); vch.from_python(ch); vch.mul(amp)
    vtt = h.Vector(); vtt.from_python(time); vtt.mul(Fs)
    return vch, vtt

# get chirp stim: based on sam's code form evoizhi/sim.py
def getChirpLog(f0, f1, t0, amp, Fs, delay):
    time = np.linspace(0,t0+delay*2, (t0+delay*2)*Fs*40+1)
    chirp_time = np.linspace(0, t0, (t0)*Fs*40+1)
    ch = chirp(chirp_time, f0, t0, f1, method='log',phi=-90)
    ch = np.hstack((np.zeros(Fs*40*delay), ch, np.zeros(Fs*40*delay)))
    vch = h.Vector(); vch.from_python(ch); vch.mul(amp)
    vtt = h.Vector(); vtt.from_python(time); vtt.mul(Fs)
    return vch, vtt

#calculate path length between two sections
def fromtodistance(origin_segment, to_segment):
    h.distance(0, origin_segment.x, sec=origin_segment.sec)
    return h.distance(to_segment.x, sec=to_segment.sec)

# comput impedance measures
def zMeasures(current, v,  delay, sampr, f1, bwinsz=1):
    ## zero padding
    current = current[int(delay*sampr - 0.5*sampr+1):-int(delay*sampr- 0.5*sampr)]
    current = np.hstack((np.repeat(current[0],int(delay*sampr)),current, np.repeat(current[-1], int(delay*sampr))))
    current = current - np.mean(current)
    v = v[int(delay*sampr - 0.5*sampr)+1:-int(delay*sampr - 0.5*sampr)]
    v = np.hstack((np.repeat(v[0],int(delay*sampr)), v, np.repeat(v[-1], int(delay*sampr))))
    v = v - np.mean(v)

    ## input and transfer impedance
    f_current = (fft(current)/len(current))[0:int(len(current)/2)]
    f_cis = (fft(v)/len(v))[0:int(len(v)/2)]
    z = f_cis / f_current

    ## impedance measures
    Freq       = np.linspace(0.0, sampr/2.0, len(z))
    zAmp       = abs(z)
    zPhase     = np.arctan2(np.imag(z),np.real(z))
    zRes       = np.real(z)
    zReact     = np.imag(z)

    ## smoothing
    fblur = np.array([1.0/bwinsz for i in range(bwinsz)])
    zAmp = convolve(zAmp,fblur,'same')
    zPhase = convolve(zPhase, fblur, 'same')

    ## trim
    mask = (Freq >= 0.5) & (Freq <= f1)
    Freq, zAmp, zPhase, zRes, zReact, z = Freq[mask], zAmp[mask], zPhase[mask], zRes[mask], zReact[mask], z[mask]

    ## resonance
    zResAmp    = np.max(zAmp)
    zResFreq   = Freq[np.argmax(zAmp)]
    Qfactor    = zResAmp / zAmp[0]
    fVar       = np.std(zAmp) / np.mean(zAmp)

    peak_to_peak = np.max(v) - np.min(v)

    return Freq, zAmp, zPhase, zRes, zReact, zResAmp, zResFreq, Qfactor, fVar, peak_to_peak, z

# voltage attenuation
def Vattenuation(ZinAmp, ZcAmp):
    out = (ZinAmp-ZcAmp) / ZinAmp
    return out

# phase lag
def phaseLag(ZinPhase, ZcPhase):
    out = ZinPhase - ZcPhase
    return out

# apply chirp stimulus to segment
def applyChirp(I, t, seg, soma_seg, t0, delay, Fs, f1, out_file_name = None):

    ## place current clamp on soma
    stim = h.IClamp(seg)
    stim.amp = 0
    stim.dur = (t0+delay*2) * Fs + 1
    I.play(stim._ref_amp, t)

    ## Record time
    t_vec = h.Vector()
    t_vec.record(h._ref_t)

    ## Record soma voltage
    soma_v = h.Vector()
    soma_v.record(soma_seg._ref_v)
    cis_v = h.Vector()
    cis_v.record(seg._ref_v)
    
    ## run simulation
    h.celsius = 34
    h.tstop = (t0+delay*2) * Fs + 1
    h.run()

    soma_np = soma_v.as_numpy()
    
    current_np = np.interp(np.linspace(0, (t0+delay*2) * Fs, soma_np.shape[0], endpoint=True),
                           np.linspace(0,(t0+delay*2) * Fs,(t0+delay*2) * Fs * 40 + 1,endpoint=True), I.as_numpy())
    time = t_vec.as_numpy()
    cis_np = cis_v.as_numpy()
    
    samp_rate = (1 / (time[1] - time[0])) * Fs
    
    ## calculate impedance
    Freq, ZinAmp, ZinPhase, ZinRes, ZinReact, ZinResAmp, ZinResFreq, QfactorIn, fVarIn, peak_to_peak, z = zMeasures(current_np, cis_np,  delay, samp_rate, f1, bwinsz=5)
    _, ZcAmp, ZcPhase, ZcRes, ZcReact, ZcResAmp, ZcResFreq, QfactorTrans, fVarTrans, peak_to_peak, z = zMeasures(current_np, soma_np,  delay, samp_rate, f1, bwinsz=5)

    freqsIn = np.argwhere(ZinPhase > 0)
    if len(freqsIn) > 0:
        ZinSynchFreq = Freq[freqsIn[-1]]
        ZinPhaseL = np.trapz([float(ZinPhase[ind]) for ind in freqsIn], 
            [float(Freq[ind]) for ind in freqsIn])
    else:
        ZinSynchFreq = 0 
        ZinPhaseL = 0

    freqsC = np.argwhere(ZcPhase > 0)
    if len(freqsC) > 0:
        ZcSynchFreq = Freq[freqsC[-1]]
        ZcPhaseL = np.trapz([float(ZcPhase[ind]) for ind in freqsC], 
            [float(Freq[ind]) for ind in freqsC])
    else:
        ZcSynchFreq = 0
        ZcPhaseL = 0

    v_attenuation = Vattenuation(ZinAmp, ZcAmp)
    phase_lag = phaseLag(ZinPhase, ZcPhase)

    dist = fromtodistance(seg, soma_seg)

    ## generate output
    out = {'Freq' : Freq,
        'ZinRes' : ZinRes,
        'ZinReact' : ZinReact,
        'ZinAmp' : ZinAmp,
        'ZinPhase' : ZinPhase,
        'ZcRes' : ZcRes,
        'ZcReact' : ZcReact,
        'ZcAmp' : ZcAmp,
        'ZcPhase' : ZcPhase,
        'ZinSynchFreq' : ZinSynchFreq,
        'ZinPhaseL' : ZinPhaseL,
        'ZcSynchFreq' : ZcSynchFreq,
        'ZcPhaseL' : ZcPhaseL,
        'phase_lag' : phase_lag,
        'Vattenuation' : v_attenuation,
        'ZinResAmp' : ZinResAmp,
        'ZinResFreq' : ZinResFreq,
        'ZcResAmp' : ZcResAmp,
        'ZcResFreq' : ZcResFreq,
        'QfactorIn' : QfactorIn,
        'QfactorTrans' : QfactorTrans,
        'fVarIn' : fVarIn,
        'fVarTrans' : fVarTrans,
        'dist' : dist}

    out2 = {'soma_np' : soma_np,
            'cis_np' : cis_np,
            'time' : time,
            'current_np' : current_np}

    if out_file_name:
        savemat(out_file_name + '.mat', out)
        savemat(out_file_name + '_traces.mat', out2)
    else:
        return out

# apply chirp stimulus to segment
def applyChirpHighFs(I, t, seg, soma_seg, t0, delay, Fs, f1, out_file_name = None):
    h.dt = 0.005
    ## place current clamp on soma
    stim = h.IClamp(seg)
    stim.amp = 0
    stim.dur = (t0+delay*2) * Fs + 1
    I.play(stim._ref_amp, t)

    ## Record time
    t_vec = h.Vector()
    t_vec.record(h._ref_t)

    ## Record soma voltage
    soma_v = h.Vector()
    soma_v.record(soma_seg._ref_v)
    cis_v = h.Vector()
    cis_v.record(seg._ref_v)
    
    ## run simulation
    h.celsius = 34
    h.tstop = (t0+delay*2) * Fs + 1
    h.run()

    soma_np = soma_v.as_numpy()
    
    current_np = np.interp(np.linspace(0, (t0+delay*2) * Fs, soma_np.shape[0], endpoint=True),
                           np.linspace(0,(t0+delay*2) * Fs,(t0+delay*2) * Fs *500 + 1,endpoint=True), I.as_numpy())
    time = t_vec.as_numpy()
    cis_np = cis_v.as_numpy()
    
    samp_rate = (1 / (time[1] - time[0])) * Fs
    
    ## calculate impedance
    Freq, ZinAmp, ZinPhase, ZinRes, ZinReact, ZinResAmp, ZinResFreq, QfactorIn, fVarIn, peak_to_peak = zMeasures(current_np, cis_np,  delay, samp_rate, f1, bwinsz=5)
    _, ZcAmp, ZcPhase, ZcRes, ZcReact, ZcResAmp, ZcResFreq, QfactorTrans, fVarTrans, peak_to_peak = zMeasures(current_np, soma_np,  delay, samp_rate, f1, bwinsz=5)

    freqsIn = np.argwhere(ZinPhase > 0)
    if len(freqsIn) > 0:
        ZinSynchFreq = Freq[freqsIn[-1]]
        ZinPhaseL = np.trapz([float(ZinPhase[ind]) for ind in freqsIn], 
            [float(Freq[ind]) for ind in freqsIn])
    else:
        ZinSynchFreq = 0 
        ZinPhaseL = 0

    freqsC = np.argwhere(ZcPhase > 0)
    if len(freqsC) > 0:
        ZcSynchFreq = Freq[freqsC[-1]]
        ZcPhaseL = np.trapz([float(ZcPhase[ind]) for ind in freqsC], 
            [float(Freq[ind]) for ind in freqsC])
    else:
        ZcSynchFreq = 0
        ZcPhaseL = 0

    v_attenuation = Vattenuation(ZinAmp, ZcAmp)
    phase_lag = phaseLag(ZinPhase, ZcPhase)

    dist = fromtodistance(seg, soma_seg)

    ## generate output
    out = {'Freq' : Freq,
        'ZinRes' : ZinRes,
        'ZinReact' : ZinReact,
        'ZinAmp' : ZinAmp,
        'ZinPhase' : ZinPhase,
        'ZcRes' : ZcRes,
        'ZcReact' : ZcReact,
        'ZcAmp' : ZcAmp,
        'ZcPhase' : ZcPhase,
        'ZinSynchFreq' : ZinSynchFreq,
        'ZinPhaseL' : ZinPhaseL,
        'ZcSynchFreq' : ZcSynchFreq,
        'ZcPhaseL' : ZcPhaseL,
        'phase_lag' : phase_lag,
        'Vattenuation' : v_attenuation,
        'ZinResAmp' : ZinResAmp,
        'ZinResFreq' : ZinResFreq,
        'ZcResAmp' : ZcResAmp,
        'ZcResFreq' : ZcResFreq,
        'QfactorIn' : QfactorIn,
        'QfactorTrans' : QfactorTrans,
        'fVarIn' : fVarIn,
        'fVarTrans' : fVarTrans,
        'dist' : dist}

    out2 = {'soma_np' : soma_np,
            'cis_np' : cis_np,
            'time' : time,
            'current_np' : current_np}

    if out_file_name:
        savemat(out_file_name + '.mat', out)
        savemat(out_file_name + '_traces.mat', out2)
    else:
        return out

# apply chirp stimulus to segment
def applyChirpVarDt(I, t, seg, soma_seg, t0, delay, Fs, f1, out_file_name = None):

    ## place current clamp on soma
    stim = h.IClamp(seg)
    stim.amp = 0
    stim.dur = (t0+delay*2) * Fs + 1
    I.play(stim._ref_amp, t)

    # ## Record time
    # t_vec = h.Vector()
    # t_vec.record(h._ref_t)
    t_vec = h.CVode().record(h._ref_t)

    ## Record soma voltage
    soma_v = h.Vector()
    # soma_v.record(soma_seg._ref_v)
    cis_v = h.Vector()
    # cis_v.record(seg._ref_v)
    h.CVode().record(soma_seg._ref_v, soma_v, t_vec)
    h.CVode().record(seg._ref_v, cis_v, t_vec)
    
    ## run simulation
    h.celsius = 34
    h.tstop = (t0+delay*2) * Fs + 1
    h.CVode()
    while h.t < h.tstop:
        h.fadvance()
        h.t = h.t + h.dt 

    soma_np = soma_v.as_numpy()
    
    current_np = np.interp(np.linspace(0, (t0+delay*2) * Fs, soma_np.shape[0], endpoint=True),
                           np.linspace(0,(t0+delay*2) * Fs,(t0+delay*2) * Fs * 40 + 1,endpoint=True), I.as_numpy())
    time = t_vec.as_numpy()
    cis_np = cis_v.as_numpy()
    
    samp_rate = (1 / (time[1] - time[0])) * Fs
    
    ## calculate impedance
    Freq, ZinAmp, ZinPhase, ZinRes, ZinReact, ZinResAmp, ZinResFreq, QfactorIn, fVarIn, peak_to_peak = zMeasures(current_np, cis_np,  delay, samp_rate, f1, bwinsz=5)
    _, ZcAmp, ZcPhase, ZcRes, ZcReact, ZcResAmp, ZcResFreq, QfactorTrans, fVarTrans, peak_to_peak = zMeasures(current_np, soma_np,  delay, samp_rate, f1, bwinsz=5)

    freqsIn = np.argwhere(ZinPhase > 0)
    if len(freqsIn) > 0:
        ZinSynchFreq = Freq[freqsIn[-1]]
        ZinPhaseL = np.trapz([float(ZinPhase[ind]) for ind in freqsIn], 
            [float(Freq[ind]) for ind in freqsIn])
    else:
        ZinSynchFreq = 0 
        ZinPhaseL = 0

    freqsC = np.argwhere(ZcPhase > 0)
    if len(freqsC) > 0:
        ZcSynchFreq = Freq[freqsC[-1]]
        ZcPhaseL = np.trapz([float(ZcPhase[ind]) for ind in freqsC], 
            [float(Freq[ind]) for ind in freqsC])
    else:
        ZcSynchFreq = 0
        ZcPhaseL = 0

    v_attenuation = Vattenuation(ZinAmp, ZcAmp)
    phase_lag = phaseLag(ZinPhase, ZcPhase)

    dist = fromtodistance(seg, soma_seg)

    ## generate output
    out = {'Freq' : Freq,
        'ZinRes' : ZinRes,
        'ZinReact' : ZinReact,
        'ZinAmp' : ZinAmp,
        'ZinPhase' : ZinPhase,
        'ZcRes' : ZcRes,
        'ZcReact' : ZcReact,
        'ZcAmp' : ZcAmp,
        'ZcPhase' : ZcPhase,
        'ZinSynchFreq' : ZinSynchFreq,
        'ZinPhaseL' : ZinPhaseL,
        'ZcSynchFreq' : ZcSynchFreq,
        'ZcPhaseL' : ZcPhaseL,
        'phase_lag' : phase_lag,
        'Vattenuation' : v_attenuation,
        'ZinResAmp' : ZinResAmp,
        'ZinResFreq' : ZinResFreq,
        'ZcResAmp' : ZcResAmp,
        'ZcResFreq' : ZcResFreq,
        'QfactorIn' : QfactorIn,
        'QfactorTrans' : QfactorTrans,
        'fVarIn' : fVarIn,
        'fVarTrans' : fVarTrans,
        'dist' : dist}

    out2 = {'soma_np' : soma_np,
            'cis_np' : cis_np,
            'time' : time,
            'current_np' : current_np}

    if out_file_name:
        savemat(out_file_name + '.mat', out)
        savemat(out_file_name + '_traces.mat', out2)
    else:
        return out

def applyChirpPlusNoise(Ichirp, Inoise, t, tnoise, seg, t0, delay, Fs, f1, out_file_name = None):
    ## place current clamp on soma
    stim = h.IClamp(seg)
    stim.amp = 0
    stim.dur = (t0+delay*2) * Fs + 1
    Ichirp.play(stim._ref_amp, t)

    stimNoise = h.IClamp(seg)
    stimNoise.amp = 0
    stimNoise.dur = (t0+delay*2) * Fs + 1
    Inoise.play(stimNoise._ref_amp, tnoise)

    ## Record time
    t_vec = h.Vector()
    t_vec.record(h._ref_t)

    ## Record soma voltage
    cis_v = h.Vector()
    cis_v.record(seg._ref_v)
    
    ## run simulation
    h.celsius = 34
    h.tstop = (t0+delay*2) * Fs + 1
    h.run()

    time = t_vec.as_numpy()
    cis_np = cis_v.as_numpy()
    current_np = np.interp(np.linspace(0, (t0+delay*2) * Fs, cis_np.shape[0], endpoint=True),
                           np.linspace(0,(t0+delay*2) * Fs,(t0+delay*2) * Fs * 40 + 1,endpoint=True), Ichirp.as_numpy())
    
    samp_rate = (1 / (time[1] - time[0])) * Fs

    ## calculate impedance
    Freq, ZinAmp, ZinPhase, ZinRes, ZinReact, ZinResAmp, ZinResFreq, QfactorIn, fVarIn, peak_to_peak = zMeasures(current_np, cis_np,  delay, samp_rate, f1, bwinsz=1)

    freqsIn = np.argwhere(ZinPhase > 0)
    if len(freqsIn) > 0:
        ZinSynchFreq = Freq[freqsIn[-1]]
        ZinPhaseL = np.trapz([float(ZinPhase[ind]) for ind in freqsIn], 
            [float(Freq[ind]) for ind in freqsIn])
    else:
        ZinSynchFreq = 0 
        ZinPhaseL = 0

    pre_v = [cis_v for cis_v, T in zip(cis_np, time) if (delay-1)*1000 <= T <= delay*1000]
    noise_peak_to_peak = np.max(pre_v) - np.min(pre_v)

    ## generate output
    out = {'Freq' : Freq,
        'ZinRes' : ZinRes,
        'ZinReact' : ZinReact,
        'ZinAmp' : ZinAmp,
        'ZinPhase' : ZinPhase,
        'ZinSynchFreq' : ZinSynchFreq,
        'ZinPhaseL' : ZinPhaseL,
        'ZinResAmp' : ZinResAmp,
        'ZinResFreq' : ZinResFreq,
        'QfactorIn' : QfactorIn,
        'fVarIn' : fVarIn,
        'noise_peak_to_peak' : noise_peak_to_peak,
        'chirp_peak_to_peak' : peak_to_peak}

    out2 = {'cis_np' : cis_np,
            'time' : time,
            'current_np' : current_np}

    if out_file_name:
        savemat(out_file_name + '.mat', out)
        savemat(out_file_name + '_traces.mat', out2)
    else:
        return out

# apply chirp stimulus to segment
def applyChirpZin(I, t, seg, t0, delay, Fs, f1, out_file_name = None):

    ## place current clamp on soma
    stim = h.IClamp(seg)
    stim.amp = 0
    stim.dur = (t0+delay*2) * Fs + 1
    I.play(stim._ref_amp, t)

    ## Record time
    t_vec = h.Vector()
    t_vec.record(h._ref_t)

    ## Record soma voltage
    cis_v = h.Vector()
    cis_v.record(seg._ref_v)
    
    ## run simulation
    h.celsius = 34
    h.tstop = (t0+delay*2) * Fs + 1
    h.run()

    time = t_vec.as_numpy()
    cis_np = cis_v.as_numpy()
    current_np = np.interp(np.linspace(0, (t0+delay*2) * Fs, cis_np.shape[0], endpoint=True),
                           np.linspace(0,(t0+delay*2) * Fs,(t0+delay*2) * Fs * 40 + 1,endpoint=True), I.as_numpy())
    
    samp_rate = (1 / (time[1] - time[0])) * Fs
    
    ## calculate impedance
    Freq, ZinAmp, ZinPhase, ZinRes, ZinReact, ZinResAmp, ZinResFreq, QfactorIn, fVarIn, peak_to_peak, z = zMeasures(current_np, cis_np,  delay, samp_rate, f1, bwinsz=1)

    freqsIn = np.argwhere(ZinPhase > 0)
    if len(freqsIn) > 0:
        ZinSynchFreq = Freq[freqsIn[-1]]
        ZinPhaseL = np.trapz([float(ZinPhase[ind]) for ind in freqsIn], 
            [float(Freq[ind]) for ind in freqsIn])
    else:
        ZinSynchFreq = 0 
        ZinPhaseL = 0

    ## generate output
    out = {'Freq' : Freq,
        'ZinRes' : ZinRes,
        'ZinReact' : ZinReact,
        'ZinAmp' : ZinAmp,
        'ZinPhase' : ZinPhase,
        'ZinSynchFreq' : ZinSynchFreq,
        'ZinPhaseL' : ZinPhaseL,
        'ZinResAmp' : ZinResAmp,
        'ZinResFreq' : ZinResFreq,
        'QfactorIn' : QfactorIn,
        'fVarIn' : fVarIn,
        'z' : z}

    out2 = {'cis_np' : cis_np,
            'time' : time,
            'current_np' : current_np}

    if out_file_name:
        savemat(out_file_name + '.mat', out)
        savemat(out_file_name + '_traces.mat', out2)
    else:
        return out

# setup gaussian white noise for STA
def getNoise(avg, std, t0, amp, Fs, delay):
    time = np.linspace(0,t0+delay*2, (t0+delay*2)*Fs+1)
    means = [avg for i in range(int(t0*Fs+1))]
    stds = np.repeat(std,t0*Fs+1)
    ch = np.random.normal(means, stds, (t0)*Fs+1)
    ch = np.hstack((np.add(np.zeros(Fs*delay),avg), ch, np.add(np.zeros(Fs*delay),avg)))
    vch = h.Vector(); vch.from_python(ch); vch.mul(amp)
    vtt = h.Vector(); vtt.from_python(time); vtt.mul(Fs)
    return vch, vtt

def STA(pks, I, V, sampr, delay):
    currents = []
    voltages = []
    out = {}
    for i in range(len(pks)):
        if i == 0:
            if pks[i] > sampr * (delay+1):
                currents.append(I[int(pks[i]-sampr) : int(pks[i])])
                voltages.append(V[int(pks[i]-sampr) : int(pks[i])])
        else:
            if (pks[i]-pks[i-1]) > sampr:
                currents.append(I[int(pks[i]-sampr) : int(pks[i])])
                voltages.append(V[int(pks[i]-sampr) : int(pks[i])])
    if len(currents):
        currents = np.array(currents)
        avgI = np.mean(currents, axis=0)
        f_current = (fft(avgI)/len(avgI))[0:int(len(avgI)/2)]
        Freq = np.linspace(0.0, sampr/2.0, len(f_current))
        out = {'currents' : currents, 'voltages' : voltages, 'f_current' : f_current, 'Freq' : Freq} 
    return out

# run STA sims
def applyNoise(I, t, seg, soma_seg, t0, delay, Fs, out_file_name=None, binsize=1):
    ## place current clamp on soma
    stim = h.IClamp(seg)
    stim.amp = 0
    stim.dur = (t0+delay*2) * Fs + 1
    I.play(stim._ref_amp, t)

    ## Record time
    t_vec = h.Vector()
    t_vec.record(h._ref_t)

    ## Record soma voltage
    soma_v = h.Vector()
    soma_v.record(soma_seg._ref_v)
    cis_v = h.Vector()
    cis_v.record(seg._ref_v)
    
    ## run simulation
    h.celsius = 34
    h.tstop = (t0+delay*2) * Fs + 1
    h.run()

    soma_np = soma_v.as_numpy()
        
    current_np = np.interp(np.linspace(0, (t0+delay*2) * Fs, soma_np.shape[0], endpoint=True),
                        np.linspace(0,(t0+delay*2) * Fs,I.as_numpy().shape[0],endpoint=True), I.as_numpy())
    time = t_vec.as_numpy()
    cis_np = cis_v.as_numpy()
        
    samp_rate = (1 / (time[1] - time[0])) * Fs
        
    ## calculate impedance
    f1 = 500
    Freq, ZinAmp, ZinPhase, ZinRes, ZinReact, ZinResAmp, ZinResFreq, QfactorIn, fVarIn, peak_to_peak, zin = zMeasures(current_np, cis_np,  delay, samp_rate, f1, bwinsz=binsize)
    _, ZcAmp, ZcPhase, ZcRes, ZcReact, ZcResAmp, ZcResFreq, QfactorTrans, fVarTrans, peak_to_peak, zc = zMeasures(current_np, soma_np,  delay, samp_rate, f1, bwinsz=binsize)

    freqsIn = np.argwhere(ZinPhase > 0)
    if len(freqsIn) > 0:
        ZinSynchFreq = Freq[freqsIn[-1]]
        ZinPhaseL = np.trapz([float(ZinPhase[ind]) for ind in freqsIn], 
            [float(Freq[ind]) for ind in freqsIn])
    else:
        ZinSynchFreq = 0 
        ZinPhaseL = 0

    freqsC = np.argwhere(ZcPhase > 0)
    if len(freqsC) > 0:
        ZcSynchFreq = Freq[freqsC[-1]]
        ZcPhaseL = np.trapz([float(ZcPhase[ind]) for ind in freqsC], 
            [float(Freq[ind]) for ind in freqsC])
    else:
        ZcSynchFreq = 0
        ZcPhaseL = 0

    v_attenuation = Vattenuation(ZinAmp, ZcAmp)
    phase_lag = phaseLag(ZinPhase, ZcPhase)

    dist = fromtodistance(seg, soma_seg)

    ## generate output
    out = {'Freq' : Freq,
        'ZinRes' : ZinRes,
        'ZinReact' : ZinReact,
        'ZinAmp' : ZinAmp,
        'ZinPhase' : ZinPhase,
        'ZcRes' : ZcRes,
        'ZcReact' : ZcReact,
        'ZcAmp' : ZcAmp,
        'ZcPhase' : ZcPhase,
        'ZinSynchFreq' : ZinSynchFreq,
        'ZinPhaseL' : ZinPhaseL,
        'ZcSynchFreq' : ZcSynchFreq,
        'ZcPhaseL' : ZcPhaseL,
        'phase_lag' : phase_lag,
        'Vattenuation' : v_attenuation,
        'ZinResAmp' : ZinResAmp,
        'ZinResFreq' : ZinResFreq,
        'ZcResAmp' : ZcResAmp,
        'ZcResFreq' : ZcResFreq,
        'QfactorIn' : QfactorIn,
        'QfactorTrans' : QfactorTrans,
        'fVarIn' : fVarIn,
        'fVarTrans' : fVarTrans,
        'dist' : dist,
        'zin' : zin,
        'zc' : zc}

    out2 = {'soma_np' : soma_np,
            'cis_np' : cis_np,
            'time' : time,
            'current_np' : current_np}

    if out_file_name:
        savemat(out_file_name + '.mat', out)
        savemat(out_file_name + '_traces.mat', out2)
    else:
        return out

# compute number of bifurcations in cell morph or in section list
def computeBranchPoints(secList = None):
    N = 0
    pos = []
    if not secList:
        for sec in h.allsec():
            if len(sec.children()) == 2:
                N = N + 1
                x, y, z = sec.x3d(sec.n3d()-1), sec.y3d(sec.n3d()-1), sec.y3d(sec.n3d()-1)
                pos.append((x,y,z))
            # else:
            #     print(str(sec) + ' ' + str(len(sec.children())))
    else:
        for sec in secList:
            if len(sec.children()) == 2:
                N = N + 1
                x, y, z = sec.x3d(sec.n3d()-1), sec.y3d(sec.n3d()-1), sec.y3d(sec.n3d()-1)
                pos.append((x,y,z))
    return N, pos

def maxSegDist(soma_seg, secList = None):
    out = {'secs' : [], 'dists' : []}
    if not secList:
        for sec in h.allsec():
            for seg in sec.allseg():
                out['secs'].append(sec)
                out['dists'].append(fromtodistance(soma_seg,seg))
    ind = np.argmax(out['dists'])
    tipseg = out['secs'][ind](1)
    maxDist = fromtodistance(soma_seg, tipseg)
    return maxDist