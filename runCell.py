from geom import *
from neuron import h 
h.load_file('stdrun.hoc')
cell = PVC(0,0,0,0)
h.psection()
# cell = PyrAdr(0,0,0,0)

# secs = [cell.Adend1, cell.Adend2, cell.Adend3, cell.Bdend, cell.soma]
# for sec in secs:
#     print(str(sec))
#     h.psection(sec=sec)
