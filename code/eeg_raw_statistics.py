import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy
import mne


def calculate_percental_differencex():

    index = ["Base", "AirBnb", "Booking", "Momondo", "IAPS"]

    # TP2
    #faa_data_raw = [0.109514087, 0.057251110, 0.071358078, 0.179799688, 0.13708845]

    #val_data_raw = [-1.248776654, -0.630877430, -0.612095431, -1.065443147, -0.271450507]

    #aro_data_raw = [0.514775516, 0.414426418, 0.535997667, 0.610481994, 0.359254254]


    # TP3
    #faa_data_raw = [0.129847555, 0.092312973, -0.133734336, -0.157737070, 0.124503651]

    #val_data_raw = [-0.099689239, -0.411558783, -0.270293613, -0.366782491, -0.315933845]

    #aro_data_raw = [0.456669843, 0.414231039, 0.309219893, 0.428180811, 0.464588194]


    # TP4
    #faa_data_raw = [-0.025341022, -0.136175716, -0.055183364, -0.213782998, -0.084110115]

    #val_data_raw = [0.161199397, -0.034829776, -0.115737749, -0.199888918, 0.039891928]

    #aro_data_raw = [0.370988078, 0.393637468, 0.334873330, 0.384266104, 0.402714323]


    # TP5
    #faa_data_raw = [-0.092844193, -0.072782564, 0.006587936, -0.043288847, -0.047839630]

    #val_data_raw = [0.359026144, 0.033407724, -0.115730479, 0.162708457, 0.200348014]

    #aro_data_raw = [0.313400470, 0.389035163, 0.322428331, 0.373294564, 0.396451088]


    # TP6
    #faa_data_raw = [-0.200399958, -0.041107496, -0.111034668, -0.052136938, -0.053906760]

    #val_data_raw = [-0.085695079, 0.022630313, -0.091497096, 0.030027591, -0.000778689]

    #aro_data_raw = [0.498197716, 0.391907955, 0.374605979, 0.390469147, 0.328289921]


    # TP7
    #faa_data_raw = [0.083220760, 0.058571682, 0.048151003, 0.029108216, -0.107172888]

    #val_data_raw = [0.144858032, -0.336739087, -0.073099874, -0.457940652, -0.225275531]

    #aro_data_raw = [0.435476037, 0.414594097, 0.520437062, 0.572921278, 0.498849840]


    # TP8
    #faa_data_raw = [-0.148887490, -0.173585998, -0.000556007, -0.024346201, 0.082150209]

    #val_data_raw = [0.129736171, -0.146221042, -0.066411002, -0.019171370, 0.042083495]

    #aro_data_raw = [0.517492318, 0.435803326, 0.446835225, 0.363729260, 0.458862175]


    # TP9
    #faa_data_raw = [0.044821736, 0.075119344, -0.008558082, -0.149962745, 0.058086241]

    #val_data_raw = [0.017408636, 1.135187046, 0.583009220, 0.561656312, 0.911126799]

    #aro_data_raw = [0.381878675, 0.291108082, 0.257087808, 0.321794801, 0.223250461]

    # TP10
    #faa_data_raw = [-0.078011411, -0.082604138, -0.076710352, -0.083757255, -0.065122015]

    #val_data_raw = [0.057511526, -0.001390240, -0.129804725, 0.062407063, 0.042497201]

    #aro_data_raw = [0.323687134, 0.387321657, 0.417230502, 0.435911214, 0.442175757]


    # all
    faa_data_raw = [-0.0198, -0.0248, -0.0289, -0.0573, 0.0049]

    val_data_raw = [-0.0412, -0.0991, -0.1436, 0.0469, -0.0627]

    aro_data_raw = [0.3925, 0.3910, 0.4312, 0.3972, 0.4236]



    df_faa_data_raw = pd.DataFrame(faa_data_raw, index=index)

    df_val_data_raw = pd.DataFrame(val_data_raw, index=index)

    df_aro_data_raw = pd.DataFrame(aro_data_raw, index=index)



    faa_raw_data_percental_change = df_faa_data_raw.apply(lambda x: x.div(x.iloc[0]).subtract(1).mul(100))
    val_raw_data_percental_change = df_val_data_raw.apply(lambda x: x.div(x.iloc[0]).subtract(1).mul(100))
    aro_raw_data_percental_change = df_aro_data_raw.apply(lambda x: x.div(x.iloc[0]).subtract(1).mul(100))



    print("------", "\n\t\t\tFAA Raw:", "\t\t\t\t\t\tValenz Raw:", "\t\t\t\tArousal Raw:")
    for n in range(1, len(faa_raw_data_percental_change)):
        print(index[n], "\t\t", faa_raw_data_percental_change[0][n], "\t\t\t", val_raw_data_percental_change[0][n],
              "\t\t\t", aro_raw_data_percental_change[0][n])




if __name__ == "__main__":

    calculate_percental_differencex()
