import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import glob
import os
import matplotlib as mpl
from matplotlib import rc
from matplotlib.ticker import FormatStrFormatter

# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
# rc('text', usetex=True)
# mpl.rcParams['font.family'] = 'sans-serif'
# mpl.rcParams['text.usetex'] = True 
# mpl.rcParams['text.latex.preamble'] = [r'\usepackage[cm]{sfmath}']
# mpl.rcParams['font.family'] = 'sans-serif'
# mpl.rcParams['font.sans-serif'] = 'cm'
mpl.rcParams.update({'font.size': 14})

plot_type = 'final' #or 'global'
file_path = "/home/lilly/Energy_Harvester/Data/thesis/final"
file_name = []
dataframes = []
refc = 'dimgray'
line = 1.5
markersize = 1
vis_setting = [
    '-',
    '-',
    '-',
    '-',
    '-',
    '-'
]
c = [
    'indianred',
    'steelblue',
    'burlywood'
]
end = None
if plot_type == 'elow':

    x = 2
    y= 2
    labels = ['FF', 'PID + FF', 'PID']
    start = 400
    end = 2100
    sample = 10
    handels = [
        'g (deg)',
        'I_measured (A)',
        # 'PT1_measured (psi)',
        # 'q_measured (GPM)',
        'PT2_measured (psi)',
        'w_measured (RPM)',
    ]

    yaxis = [
        'Guide Vane Angle $g$ (deg)',
        'Current $I$ (A)',
        # 'Upstream pressure $h_1$ (kPa)',
        # 'Flow rate $q$ (L/s)',
        'Downstream pressure $h_2$ (kPa)',
        'Angular Speed $\omega$ (rad/s)'
    ]

if plot_type == 'eglobal':

    x = 2
    y = 2

    c = [
    'burlywood',
    'indianred',
    'steelblue',
    ]

    labels = ['Constant $\omega_{ref}$','OSC','FLHCS',]
    start = 300
    sample = 10
    end = 2500
    handels = [
        'g (deg)',
        'I_measured (A)',
        # 'PT1_measured (psi)',
        # 'q_measured (GPM)',
        'w_measured (RPM)',
        'P_e (W)',
    ]

    yaxis = [
        'Guide Vane Angle $g$ (deg)',
        'Current $I$ (A)',
        # 'Upstream pressure $h_1$ (kPa)',
        # 'Flow rate $q$ (L/s)',
        'Angular Speed $\omega$ (rad/s)',
        'Power $P$ (w)'
    ]

if plot_type == 'low':

    x = 2
    y= 2
    labels = ['FF','FF+PID','PID',]
    start = 400
    sample = 10
    handels = [
        'g (deg)',
        'I (A)',
        'PT2_measured (psi)',
        'w_measured (RPM)',
    ]

    yaxis = [
        'Guide Vane Angle $g$ (deg)',
        'Current $I$ (A)',
        'Pressure $h_2$ (kPa)',
        'Angular Speed $\omega$ (rad/s)'
    ]

if plot_type == 'global':

    x = 1
    y = 2

    labels = ['FLHCS','OSC','OSC + UKF',]
    start = 1000
    sample = 40
    handels = [
        'w_measured (RPM)',
        'P_e (W)',
    ]

    yaxis = [
        'Angular Speed $\omega$ (rad/s)',
        'Power $P$ (w)'
    ]

if plot_type == 'observer':

    x = 3
    y = 2
    line = 1.5
    markersize = 3
    labels = [
        ['$q$','$q$ Pred'],
        ['$\omega$','$\omega$ Pred'],
        ['$q$','$q$ Pred'],
        ['$\omega$','$\omega$ Pred'],
        ['$q$','$q$ Pred'],
        ['$\omega$','$\omega$ Pred'],
            ]
    titles = ['Day 1', 'Day 2', 'Day 3']#['Identifyied Model', 'CFD 2% Gap', 'CFD 4% Gap']
    c = [
        'indianred',
        'steelblue',
        'burlywood'
    ]
    start = 0
    end = 2500
    sample = 2
    handels = [
        ['q_measured (GPM)', 'q_pred (GPM)'],
        ['w_measured (RPM)', 'w_pred (RPM)'],
        ['q_measured (GPM)', 'q_pred (GPM)'],
        ['w_measured (RPM)', 'w_pred (RPM)'],
        ['q_measured (GPM)', 'q_pred (GPM)'],
        ['w_measured (RPM)', 'w_pred (RPM)'],
    ]
    vis_setting = [
        '-',
        '-',
        '-',
        '-'
    ]
    yaxis = [
        'Flow Rate $q$ (L/s)',
        'Angular Speed $\omega$ (rad/s)',
        'Flow Rate $q$ (L/s)',
        'Angular Speed $\omega$ (rad/s)',
        'Flow Rate $q$ (L/s)',
        'Angular Speed $\omega$ (rad/s)',
    ]

if plot_type == 'filter':

    x = 2
    y = 2
    line = 1.5
    markersize = 3
    labels = [
        ['$h_1$','$h_{1, f}$'],
        ['$V$','$V_f$ '],
        ['$q$','$q_f$'],
        ['$\omega$','$\omega_f$'],
            ]
    c = [
        'r',
        'b',
        'burlywood'
    ]
    start = 1500
    sample = 4
    end = 3000
    handels = [
        ['PT1 (psi)', 'PT1_measured (psi)'],
        ['V (V)', 'V_measured (V)'],
        ['q (GPM)', 'q_measured (GPM)'],
        ['w (RPM)', 'w_measured (RPM)'],
    ]
    vis_setting = [
        '-',
        '-',
        '-',
        '-'
    ]
    yaxis = [
        'Upstream pressure $h_1$ (kPa)',
        'Voltage $V$ (V)',
        'Flow Rate $q$ (L/s)',
        'Angular Speed $\omega$ (rad/s)',
    ]

if plot_type == 'final':

    x = 3
    y = 2
    line = 1.5
    markersize = 3
    labels = [
        ['$g$'],
        ['$I$'],
        [ '$h_1$','$h_2$','$h_{2, ref}$',],
        ['$\omega$','$\omega_{ref}$'],
        ['$P_e$'],
        ['$\eta$',],
            ]
    c = [
        ['steelblue',],
        ['steelblue',],
        ['indianred','steelblue','dimgray',],
        ['steelblue','dimgray',],
        ['indianred',],
        ['steelblue',],
    ]
    start = 0
    sample = 10
    end = 4318
    handels = [
        ['g (deg)'],
        ['I_measured (A)'],
        ['PT1_measured (psi)', 'PT2_measured (psi)','PT2_ref (psi)', ],
        ['w (RPM)', 'w_ref (RPM)',],
        ['q_measured (GPM)'],
        ['eff ()'],
    ]

    yaxis = [
        'Guide Vane Angle $g$ (deg)',
        'Current $I$ (A)',
        'Pressure $h$ (kPa)',
        'Angular Speed $\omega$ (rad/s)',
        'Flow rate $q$ (L/s)',
        'Overall efficiency $\eta$',
    ]
    
all_files = glob.glob(os.path.join(file_path, "*.csv")) #make list of paths
for f in all_files:
    name = os.path.splitext(os.path.basename(f))[0]  # Getting the file name without extension
    file_name.append(name)

file_name.sort()

for f in file_name:

    dataframe = pd.read_csv(file_path + '/' + f + ".csv")

    for col in dataframe.columns: 
        if 'psi' in col:
            dataframe[col] = dataframe[col]*6.89476
        if 'RPM' in col:
            dataframe[col] = dataframe[col]*0.104719755
        if 'GPM' in col:
            dataframe[col] = dataframe[col]*0.06309
        if 'deg' in col:
            dataframe[col] = dataframe[col] * 2.13645 + 0.01* dataframe[col]**2

    if 'z' in f and 'gap2' in file_path:
        dataframe['P_e_max (W)'][:3001] =dataframe['P_e_max (W)'][:3001] - 1
        dataframe['P_e_max (W)'][3001:4001] =dataframe['P_e_max (W)'][3001:4001] - 2
        dataframe['best_w (RPM)'][:3001] =dataframe['best_w (RPM)'][:3001] - 50
        dataframe['best_w (RPM)'][3001:4001] =dataframe['best_w (RPM)'][3001:4001] - 50
        dataframe['best_w (RPM)'][4001:5601] =dataframe['best_w (RPM)'][4001:5601] - 25

    if 'z' in f and 'gap4' in file_path:
        dataframe['P_e_max (W)'][:3001] =dataframe['P_e_max (W)'][:3001] - 3
        dataframe['P_e_max (W)'][3001:4001] =dataframe['P_e_max (W)'][3001:4001] - 3
        dataframe['best_w (RPM)'][:3001] =dataframe['best_w (RPM)'][:3001] - 70
        dataframe['best_w (RPM)'][3001:4011] =dataframe['best_w (RPM)'][3001:4011] - 70
        dataframe['best_w (RPM)'][4011:4051] =dataframe['best_w (RPM)'][4011:4051] - 30
     
    if end == None:
        dataframe = dataframe.iloc[start::sample, :]
    else:
        dataframe = dataframe.iloc[start:end:sample, :]
    dataframes.append(dataframe)

    print(f)

if plot_type != 'observer':
    fig = plt.figure(tight_layout=True, figsize=(y*5, x*3.5))
    gs = gridspec.GridSpec(x, y)

    axes = []
    for i in range(x):
        for j in range(y):
            axes.append(fig.add_subplot(gs[i,j]))

    for i, ax in enumerate(axes):
        if plot_type == 'low'or plot_type == 'elow' :

            for y, data in enumerate(dataframes[:3]):
                ax.plot(data['t (sec)'], data[handels[i]], vis_setting[i], label = labels[y], color=c[y], linewidth = line, markersize = markersize)
            if i == 2:
                ax.plot(dataframes[len(dataframes)-1]['t (sec)'], dataframes[len(dataframes)-1]['PT2_ref (psi)'], '-', label = '$h_2$ Ref', color=refc,linewidth = line, markersize = markersize)

            if i ==3:
                ax.plot(dataframes[len(dataframes)-1]['t (sec)'], dataframes[len(dataframes)-1]['w_ref (RPM)'], '-', label = '$\omega$ Ref', color=refc,linewidth = line, markersize = markersize)
            
            
            
        if plot_type == 'global':
            
            for y, data in enumerate(dataframes[:len(dataframes)-1]):
                ax.plot(data['t (sec)'], data[handels[i]], vis_setting[i], label = labels[y], color=c[y], linewidth = line, markersize = markersize)

            if i == 0:
                ax.plot(dataframes[len(dataframes)-1]['t (sec)'], dataframes[len(dataframes)-1]['best_w (RPM)'], '-', label = '$\omega$ Opt', color=refc,linewidth = line, markersize = markersize)

            if i == 1:
                ax.plot(dataframes[len(dataframes)-1]['t (sec)'], dataframes[len(dataframes)-1]['P_e_max (W)'], '-', label = '$P$ Opt', color=refc,linewidth = line, markersize = markersize)

        if plot_type == 'eglobal':
            
            for y, data in enumerate(dataframes[:len(dataframes)]):
                ax.plot(data['t (sec)'], data[handels[i]], vis_setting[i], label = labels[y], color=c[y], linewidth = line, markersize = markersize)

        if plot_type == "filter" :

            data = dataframes[0]

            for s, hand in enumerate(handels[i]):
                ax.plot(data['t (sec)'], data[hand], '-', label = labels[i][s], color=c[s], linewidth = line, markersize = markersize)

        if plot_type == "final":

            data = dataframes[0]

            for s, hand in enumerate(handels[i]):
                ax.plot(data['t (sec)'], data[hand], '-', label = labels[i][s], color=c[i][s], linewidth = line, markersize = markersize)


        ax.set_ylabel(yaxis[i])
        ax.set_xlabel('Time (sec)')
        if i == 0:
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        ax.legend(fontsize = 10)
        if 'Guide'in yaxis[i]:
            ax.set_ylim([0, 20])
        if 'Current' in yaxis[i]:
            ax.set_ylim([0.8, 3.4])
        if 'Flow' in yaxis[i]:
            ax.set_ylim([1.55,1.8])
        if 'Upstream' in yaxis[i]:
            ax.set_ylim([260,320])
        if 'Angular' in yaxis[i]:
            ax.set_ylim([200,450])
        if 'Power' in yaxis[i]:
            ax.set_ylim([0,25])
        if 'Downstream' in yaxis[i]:
            ax.set_ylim([220,260])
        if "Overall" in yaxis[i]:
            ax.set_ylim([0.1,0.4])
        else:
            ax.relim()
        ax.autoscale_view()

else:
    fig, big_axes = plt.subplots( figsize=(5*y, 3*x) , nrows=3, ncols=1, sharey=True) 

    for row, big_ax in enumerate(big_axes, start=1):
        big_ax.set_title(titles[row-1], fontsize=16)

        # Turn off axis lines and ticks of the big subplot 
        # obs alpha is 0 in RGBA string!
        big_ax.tick_params(labelcolor=(1.,1.,1., 0.0), top='off', bottom='off', left='off', right='off')
        # removes the white frame
        big_ax._frameon = False

    z = 0
    for i in range(3):
        for j in range(2):
            ax = fig.add_subplot(x,y,z+1)
            data = dataframes[i]
            for s, hand in enumerate(handels[z]):
                ax.plot(data['t (sec)'], data[hand], '-', label = labels[z][s], color=c[s], linewidth = line, markersize = markersize)
            
            if z == 0 or z == 2:
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

            ax.set_ylabel(yaxis[z])
            ax.set_xlabel('Time (sec)')
            ax.legend(loc="upper right")
            z +=1

            
plt.tight_layout()
plt.show()
