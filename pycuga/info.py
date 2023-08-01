import numpy as np
from pycuda import gpuarray, autoinit
import pycuda.driver as cuda
from pycuda.tools import DeviceData
from pycuda.tools import OccupancyRecord as occupancy


ctxCUDA =""
devdata =""

def getCudaInfo():
    nDevices = cuda.Device.count()
    ndev = None
    for i in range(nDevices):
        dev = cuda.Device( i )
        print ("  Device {0}: {1}".format( i, dev.name() ))
    devNumber = 0
    if nDevices > 1:
        if ndev == None:
            devNumber = int(raw_input("Select device number: "))
        else:
            devNumber = ndev
    dev = cuda.Device( devNumber)
    cuda.Context.pop()
    ctxCUDA = dev.make_context()
    devdata = DeviceData(dev)
    print ("Using device {0}: {1}".format( devNumber, dev.name() ))

def getKernelInfo(kernel,nthreads, rt=True):
    ''' This function returns info about kernels theoretical performance, but warning is not trivial to optimize! '''
    shared=kernel.shared_size_bytes
    regs=kernel.num_regs
    local=kernel.local_size_bytes
    const=kernel.const_size_bytes
    mbpt=kernel.max_threads_per_block
    #threads =  #self.block_size_x* self.block_size_y* self.block_size_z
    occupy = occupancy(devdata, nthreads, shared_mem=shared, registers=regs)
    print ("==Kernel Memory==")
    print("""Local:        {0}
Shared:       {1}
Registers:    {2}
Const:        {3}
Max Threads/B:{4}""".format(local,shared,regs,const,mbpt))
    print ("==Occupancy==")
    print("""Blocks executed by SM: {0}
Limited by:            {1}
Warps executed by SM:  {2}
Occupancy:             {3}""".format(occupy.tb_per_mp,occupy.limited_by,occupy.warps_per_mp,occupy.occupancy))
    if rt:
        return occupy.occupancy

def gpuMesureTime(myKernel, ntimes=1000):
    start = cuda.Event()
    end = cuda.Event()
    start.record()
    for i in range(ntimes):
      myKernel()
    end.record()
    end.synchronize()
    timeGPU = start.time_till(end)*1e-3
    print ("Call the function {0} times takes in GPU {1} seconds.\n".format(ntimes,timeGPU))
    print ("{0} seconds per call".format(timeGPU/ntimes))
    return timeGPU