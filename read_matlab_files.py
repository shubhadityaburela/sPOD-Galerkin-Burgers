import numpy as np
import h5py
import os

import scipy


class data_luminosity:
    def __init__(self, matlab_file) -> None:
        self.angRes: float = matlab_file['angRes'][0, 0]  # Angular resolution (d\theta)
        self.distDelta: float = matlab_file['distDelta'][0, 0]
        self.dt: float = matlab_file['dt'][0, 0]  # Step size in time
        self.fs: int = int(matlab_file['fs'][0, 0])  # Frames per second
        self.lumPerc: float = matlab_file['lumPerc'][0, 0]
        self.lumTreshold: float = matlab_file['lumTreshold'][0, 0]
        self.m: int = int(matlab_file['m'][0, 0])  # Number of time steps
        self.nSpecific: int = matlab_file['nSpecific'][0, 0]
        self.ppmm: int = matlab_file['ppmm'][0, 0]
        self.RA: float = matlab_file['RA'][0, 0]  # Outer annulus radius
        self.radRes: float = matlab_file['radRes'][0, 0]
        self.RI: float = matlab_file['RI'][0, 0]  # Inner annulus radius
        self.rM: float = matlab_file['rM'][0, 0]
        self.rot: float = matlab_file['rot'][0, 0]
        self.xN: int = matlab_file['xN'][0, 0]  # Number of Cartesian points in x-direction
        self.yN: int = matlab_file['yN'][0, 0]  # Number of Cartesian points in y-direction

        self.annCartMat: np.ndarray = np.asarray(matlab_file['annCartMat'][()])  # Luminosity value in Cartesian grid
        self.annTreshMask: np.ndarray = np.asarray(matlab_file['annTreshMask'][()])  # Threshold masking in Cartesian grid
        self.annCartCropMat: np.ndarray = np.asarray(matlab_file['annCartCropMat'][()])  # Cropped Luminosity value in Cartesian grid
        self.annPolMat: np.ndarray = np.asarray(matlab_file['annPolMat'][()])  # Luminosity value in polar grid
        self.annPolMask: np.ndarray = np.asarray(matlab_file['annPolMask'][()])  # Masking array in the polar grid
        self.annPolCropMat: np.ndarray = np.asarray(matlab_file['annPolCropMat'][()])  # Cropped Luminosity value in polar grid

        self.delta: np.ndarray = np.asarray(matlab_file['delta'][()])
        self.dist: np.ndarray = np.asarray(matlab_file['dist'][()])

        self.lumCenter: np.ndarray = np.asarray(matlab_file['lumCenter'][()])  # Luminosity values in the polar grid across the theta domain at the center of the annulus
        self.lumInt: np.ndarray = np.asarray(matlab_file['lumInt'][()])
        self.lumMax: np.ndarray = np.asarray(matlab_file['lumMax'][()])  # Maximum luminosity value in the entire polar grid per time step
        self.lumProbeAmpMat: np.ndarray = np.asarray(matlab_file['lumProbeAmpMat'][()])
        self.lumProbeFreqMat: np.ndarray = np.asarray(matlab_file['lumProbeFreqMat'][()])


        self.meanArea: np.ndarray = np.asarray(matlab_file['meanArea'][()])
        self.meanDens: np.ndarray = np.asarray(matlab_file['meanDens'][()])
        self.meanDensMat: np.ndarray = np.asarray(matlab_file['meanDensMat'][()])
        self.r: np.ndarray = np.asarray(matlab_file['r'][()])
        self.R: np.ndarray = np.asarray(matlab_file['R'][()])
        self.stdArea: np.ndarray = np.asarray(matlab_file['stdArea'][()])
        self.stdDelta: np.ndarray = np.asarray(matlab_file['stdDelta'][()])
        self.stdDens: np.ndarray = np.asarray(matlab_file['stdDens'][()])
        self.theta: np.ndarray = np.asarray(matlab_file['theta'][()])
        self.X: np.ndarray = np.asarray(matlab_file['X'][()])  # X coordinates of the Cartesian grid
        self.Y: np.ndarray = np.asarray(matlab_file['Y'][()])  # Y coordinates of the Cartesian grid
        self.Xtilde: np.ndarray = np.asarray(matlab_file['Xtilde'][()])  # R coordinates of the polar grid
        self.Ytilde: np.ndarray = np.asarray(matlab_file['Ytilde'][()])  # \theta coordinates of the polar grid
        self.xx: np.ndarray = np.asarray(matlab_file['xx'][()])
        self.yy: np.ndarray = np.asarray(matlab_file['yy'][()])


class data_pressure:
    def __init__(self, matlab_file) -> None:
        self.dt: float = matlab_file['dt'][0, 0]  # Time step size
        self.fs: int = matlab_file['fs'][0, 0]  # Frames per second
        self.ProbePos: np.ndarray = np.asarray(matlab_file['ProbePos'][()])  # Azimuthal positions of probe placement
        self.pressureProbeAmpMat: np.ndarray = np.asarray(matlab_file['pressureProbeAmpMat'][()])  # Pressure amplitudes at the sensor positions
        self.pressureProbeFreqMat: np.ndarray = np.asarray(matlab_file['pressureProbeFreqMat'][()])  # Frequency at the sensor positions
        self.pressureProbeMat: np.ndarray = np.asarray(matlab_file['pressureProbeMat'][()])

        self.RA: float = 45.0  # Outer annulus radius



class new_raw_data_pressure:
    def __init__(self, matlab_file) -> None:
        self.fs: int = matlab_file['D']['config'][0, 0]['sampleRate'][0, 0][0, 0]
        self.t: np.ndarray = matlab_file['D']['time'][0, 0]['t'][0, 0][0]
        self.pcb01: np.ndarray = matlab_file['D']['pcb01'][0, 0]['data'][0, 0][0]
        self.pcb02: np.ndarray = matlab_file['D']['pcb02'][0, 0]['data'][0, 0][0]
        self.pcb03: np.ndarray = matlab_file['D']['pcb03'][0, 0]['data'][0, 0][0]
        self.pcb04: np.ndarray = matlab_file['D']['pcb04'][0, 0]['data'][0, 0][0]
        self.pcb05: np.ndarray = matlab_file['D']['pcb05'][0, 0]['data'][0, 0][0]
        self.pcb06: np.ndarray = matlab_file['D']['pcb06'][0, 0]['data'][0, 0][0]
        self.pcb07: np.ndarray = matlab_file['D']['pcb07'][0, 0]['data'][0, 0][0]
        self.pcb08: np.ndarray = matlab_file['D']['pcb08'][0, 0]['data'][0, 0][0]
        self.pcb09: np.ndarray = matlab_file['D']['pcb09'][0, 0]['data'][0, 0][0]
        self.pcb10: np.ndarray = matlab_file['D']['pcb10'][0, 0]['data'][0, 0][0]
        self.pcb11: np.ndarray = matlab_file['D']['pcb11'][0, 0]['data'][0, 0][0]
        self.pcb12: np.ndarray = matlab_file['D']['pcb12'][0, 0]['data'][0, 0][0]
        self.pcb13: np.ndarray = matlab_file['D']['pcb13'][0, 0]['data'][0, 0][0]
        self.pcb14: np.ndarray = matlab_file['D']['pcb14'][0, 0]['data'][0, 0][0]
        self.pcb15: np.ndarray = matlab_file['D']['pcb15'][0, 0]['data'][0, 0][0]


class new_processed_data_pressure:
    def __init__(self, matlab_file) -> None:
        self.windowed_data = matlab_file['D']['window'][0, 0]


        self.t: np.ndarray = self.windowed_data['data'][0, 0]['time'][0, 0][0]
        self.pcb01_cut: np.ndarray = self.windowed_data['data'][0, 0]['pcb01'][0, 0][0]
        self.pcb02_cut: np.ndarray = self.windowed_data['data'][0, 0]['pcb02'][0, 0][0]
        self.pcb03_cut: np.ndarray = self.windowed_data['data'][0, 0]['pcb03'][0, 0][0]
        self.pcb04_cut: np.ndarray = self.windowed_data['data'][0, 0]['pcb04'][0, 0][0]
        self.pcb05_cut: np.ndarray = self.windowed_data['data'][0, 0]['pcb05'][0, 0][0]
        self.pcb06_cut: np.ndarray = self.windowed_data['data'][0, 0]['pcb06'][0, 0][0]
        self.pcb07_cut: np.ndarray = self.windowed_data['data'][0, 0]['pcb07'][0, 0][0]
        self.pcb08_cut: np.ndarray = self.windowed_data['data'][0, 0]['pcb08'][0, 0][0]
        self.pcb09_cut: np.ndarray = self.windowed_data['data'][0, 0]['pcb09'][0, 0][0]
        self.pcb10_cut: np.ndarray = self.windowed_data['data'][0, 0]['pcb10'][0, 0][0]
        self.pcb11_cut: np.ndarray = self.windowed_data['data'][0, 0]['pcb11'][0, 0][0]
        self.pcb12_cut: np.ndarray = self.windowed_data['data'][0, 0]['pcb12'][0, 0][0]
        self.pcb13_cut: np.ndarray = self.windowed_data['data'][0, 0]['pcb13'][0, 0][0]
        self.pcb14_cut: np.ndarray = self.windowed_data['data'][0, 0]['pcb14'][0, 0][0]
        self.pcb15_cut: np.ndarray = self.windowed_data['data'][0, 0]['pcb15'][0, 0][0]


def read_matlab(dir, variable, name):
    return h5py.File(os.path.join(dir, variable, name), 'r')


def new_read_matlab(dir, folder, name):
    path = os.path.join(dir, folder, name)
    try:
        # Try reading as HDF5 (v7.3)
        return h5py.File(path, 'r')
    except OSError:
        # Fallback for v7 and below
        return scipy.io.loadmat(path)
