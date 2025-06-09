from PyEMD import EMD

def get_EMD(series):
    emd = EMD.EMD()
    return emd(series)