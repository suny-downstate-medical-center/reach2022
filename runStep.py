from geom import *
# from neuron import h 
# h.load_file('stdrun.hoc')
# cell = PVC(0,0,0,0)
# h.psection()
cell = PyrAdr(0,0,0,0)


clamp = h.IClamp(cell.Adend1(1))
clamp.delay = 500 
clamp.dur = 200 
clamp.amp = -0.08

v_soma = h.Vector().record(cell.soma(0.5)._ref_v)
time = h.Vector().record(h._ref_t)

h.tstop = 1500 
h.run()

from matplotlib import pyplot as plt
plt.ion()
v_cut = [v for v, t in zip(v_soma.as_numpy(), time.as_numpy()) if 450 < t < 750]
t_cut = [t for t in time.as_numpy() if 450 < t < 750]
# plt.plot(time, v_soma)
plt.plot(t_cut, np.subtract(v_cut, v_cut[0]))

######################################3
# adjust Ih here 
# example - reduce 50% everywhere 
# for sec in h.allsec(): # loop through all sections 
#     for seg in sec.allseg(): # loop through all segments 
#         try: # try to reduce by 50 %
#             seg.hcurrent.gbar = seg.hcurrent.gbar * 0.5
#             print(seg)
#         except: # if that doesn't work, do nothing 
#             pass 
cell.Adend3(0.5).hcurrent.gbar = cell.Adend3(0.5).hcurrent.gbar * 0.5
cell.Adend2(0.5).hcurrent.gbar = cell.Adend2(0.5).hcurrent.gbar * 0.5
cell.Adend1(0.5).hcurrent.gbar = cell.Adend1(0.5).hcurrent.gbar * 0.5


h.finitialize()
h.run()
# plt.plot(time, v_soma)
v_cut = [v for v, t in zip(v_soma.as_numpy(), time.as_numpy()) if 450 < t < 750]
t_cut = [t for t in time.as_numpy() if 450 < t < 750]
plt.plot(t_cut, np.subtract(v_cut, v_cut[0]))

# secs = [cell.Adend1, cell.Adend2, cell.Adend3, cell.Bdend, cell.soma]
# for sec in secs:
#     print(str(sec))
#     h.psection(sec=sec)
