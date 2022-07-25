from netpyne import specs 

netParams = specs.NetParams()
simConfig = specs.SimConfig()

from neuron import h 
h.load_file('stdrun.hoc')
netParams.importCellParams(label='PYR', fileName='geom.py', cellName='PyrAdr', cellArgs={'x' : 0, 'y' : 0, 'z' : 0, 'id' : 0})
netParams.importCellParams(label='OLM', fileName='geom.py', cellName='Ow', cellArgs={'x' : 0, 'y' : 0, 'z' : 0, 'id' : 0})
netParams.importCellParams(label='PV', fileName='geom.py', cellName='PVC', cellArgs={'x' : 0, 'y' : 0, 'z' : 0, 'id' : 0})

netParams.popParams['PYR'] = {'cellType' : 'PYR', 'numCells' : 1}
netParams.popParams['OLM'] = {'cellType' : 'OLM', 'numCells' : 1}
netParams.popParams['PV'] + {'cellType' : 'PV', 'numCells' : 1}