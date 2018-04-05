import pandas as pd


def calculate_percental_differences():

    # the data values for each TP according to FAA, Valenz and Arousal
    # the values are sorted as follows: baseline, Airbnb, Booking.com,
    # momondo and IAPS

    # TP2
    # faa_data_epoched = [-0.291789743, 0.060915116, 0.032066176, 0.259229482, 0.342928636]

    # val_data_epoched = [-4.286414340, 1.134730866, -3.720215542, -1.165232005, 0.868339508]

    # aro_data_epoched = [0.485794832, 0.366571927, 0.527038660, 0.824905251, 1.024694680]


    # TP3
    # faa_data_epoched = [0.143412063, 0.699762812, -0.007026621, -0.170059731, -0.311732170]

    # val_data_epoched = [-0.088105924, 2.172068325, 1.034848845, 1.453157231, -0.937642041]

    # aro_data_epoched = [5.234316741, 0.660907912, 0.373682491, 0.532277004, 0.563313436]



    # TP4
    # faa_data_epoched = [-0.596988298, -0.250969604, -0.512810164, 0.610924670, 0.280395860]

    # val_data_epoched = [-0.203914442, 1.589320167, -0.006433315, 0.056687470, -0.020494470]

    # aro_data_epoched = [5.264289148, 0.264754221, 0.703799371, 0.612274898, 0.555491009]


    # TP5
    # faa_data_epoched = [0.154509899, -0.543700739, -0.352181221, 0.316954306, 0.360579129]

    # val_data_epoched = [-0.246729522, -0.685405341, -1.021681192, 1.332739744, 0.302681281]

    # aro_data_epoched = [0.843377339, 0.555486238, 0.214220438, 0.700106812, 0.101116620]


    # TP6
    # faa_data_epoched = [0.345336695, -0.123960115, -0.210544593, 0.088400978, -0.324254218]

    # val_data_epoched = [1.896773677, 1.053210227, 1.269113301, 1.529216223, -2.824714786]

    # aro_data_epoched = [0.365467679, 0.247355916, 0.248964131, 0.801996526, 0.339150845]


    # TP7
    # faa_data_epoched = [0.193890826, 0.424333772, 0.108495671, 0.697027710, -0.143805665]

    # val_data_epoched = [-1.170658099, 2.645414620, -3.247139750, -0.223327558, -2.118089209]

    # aro_data_epoched = [1.164463149, 0.472616122, 0.343402638, 1.709583518, 0.820487498]


    # TP8
    # faa_data_epoched = [-0.407753149, 0.034745984, 0.492827810, -0.071635287, -0.068709003]

    # val_data_epoched = [-0.227958149, 0.895929423, 2.931455595, -2.039205454, -1.544773202]

    # aro_data_epoched = [1.924586498, 0.963244742, 0.537704963, 0.419642241, 0.605971625]


    # TP9
    # faa_data_epoched = [0.233843708, 0.037250839, -0.093916816, -0.130041141, -0.005305385]

    # val_data_epoched = [0.117654983, 0.722236169, -0.763805692, -1.042354261, -0.890594094]

    # aro_data_epoched = [1.079222006, 1.247942668, 0.823519765, 0.681369738, 0.529315204]


    # TP10
    # faa_data_epoched = [-0.376728410, 0.355667603, -0.744885966, 0.263336876, -0.453870588]

    # val_data_epoched = [-2.007512345, 0.280119713, 0.005895436, 1.341619842, -1.559865868]

    # aro_data_epoched = [0.492753788, 0.791159713, 2.270792675, 0.753984544, 0.198222692]



    # index for proper printing of the results
    index = ["Base", "AirBnb", "Booking", "Momondo", "IAPS"]


    # create pandas dataframes of the data for easier calculations
    df_faa_data_epoched = pd.DataFrame(faa_data_epoched, index=index)

    df_val_data_epoched = pd.DataFrame(val_data_epoched, index=index)

    df_aro_data_epoched = pd.DataFrame(aro_data_epoched, index=index)



    # calculate the percential differences for one user between all four
    # conditions concerning to the baseline-data
    faa_epoched_data_percental_change = df_faa_data_epoched.apply(lambda x: x.div(x.iloc[0]).subtract(1).mul(100))

    val_epoched_data_percental_change = df_val_data_epoched.apply(lambda x: x.div(x.iloc[0]).subtract(1).mul(100))

    aro_epoched_data_percental_change = df_aro_data_epoched.apply(lambda x: x.div(x.iloc[0]).subtract(1).mul(100))



    print("\n------", "\n\t\t\tFAA Epoched:", "\t\t\tValenz Epoched:", "\t\tArousal Epoched:")
    for n in range(1, len(faa_epoched_data_percental_change)):
        print(index[n], "\t\t", "%.2f" % faa_epoched_data_percental_change[0][n], "\t\t\t\t\t", "%.2f" % val_epoched_data_percental_change[0][n], "\t\t\t\t", "%.2f" % aro_epoched_data_percental_change[0][n])



if __name__ == "__main__":

    calculate_percental_differences()
