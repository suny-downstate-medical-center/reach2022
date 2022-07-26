from geom import *
# from neuron import h 
# h.load_file('stdrun.hoc')
# cell = PVC(0,0,0,0)
# h.psection()
cell = PyrAdr(0,0,0,0)


clamp = h.IClamp(cell.soma(0.5))
clamp.delay = 500 
clamp.dur = 500 
clamp.amp = -0.2

v_soma = h.Vector().record(cell.soma(0.5)._ref_v)
time = h.Vector().record(h._ref_t)

h.tstop = 1500 
h.run()

from matplotlib import pyplot as plt
plt.ion()
plt.plot(time, v_soma)

######################################3
# adjust Ih here 
# example - reduce 50% everywhere 
for sec in h.allsec(): # loop through all sections 
    for seg in sec.allseg(): # loop through all segments 
        try: # try to reduce by 50 %
            seg.hcurrent.gbar = seg.hcurrent.gbar * 0.5
            print(seg)
        except: # if that doesn't work, do nothing 
            pass 

h.finitialize()
h.run()
plt.plot(time, v_soma)

# secs = [cell.Adend1, cell.Adend2, cell.Adend3, cell.Bdend, cell.soma]
# for sec in secs:
#     print(str(sec))
#     h.psection(sec=sec)
