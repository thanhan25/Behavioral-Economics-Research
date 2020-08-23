'''
This function produces data tables and graphs for delta_v probability and data
sorted in bins by wins and absolute difference. The input is the cleaned data
in .csv format. To run the function while omitting the graphs, use
graph_x = False in the function parameters.
'''

# Import numpy, pandas, and pyplot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Assign spreadsheet file name, task_file
task_file = 'PriceObfuscation_Meeting_2019September25/table_task1.csv'

# Assign save destination
save_dest = 'Task 1/'


def task_data_graphs(task_file,
                     save_dest,
                     graph_1=True,
                     graph_2=True,
                     graph_3=True):

    # Create DataFrame Headers, task_head
    task_head = ['Participant ID',
                 'Trial ID',
                 'Task ID',
                 'Frame 1 Left Value',
                 'Frame 2 Left Value',
                 'Frame 3 Left Value',
                 'Frame 4 Left Value',
                 'Frame 5 Left Value',
                 'Frame 6 Left Value',
                 'Frame 1 Right Value',
                 'Frame 2 Right Value',
                 'Frame 3 Right Value',
                 'Frame 4 Right Value',
                 'Frame 5 Right Value',
                 'Frame 6 Right Value',
                 'Unused 1',
                 'Unused 2',
                 'Fixation Cross',
                 'Maximum Value',
                 'Minimum Value',
                 'Response Time (s)',
                 'Left Chosen',
                 'Right Chosen',
                 'Unused 3',
                 'Victory Dummy',
                 'Best Action Dummy']

    # Load CSV into Pandas DataFrame, task_data
    task_data = pd.read_csv(task_file, header=None)

    # Insert DataFrame Headers for Task1 Data
    task_data.rename(columns=dict(zip(task_data.columns[:], task_head)),
                     inplace=True)

    # Prepare data for Figure 3
    # Sum the left choice values, sum_left
    task_data['sum_left'] = task_data.iloc[:, 3:9].sum(axis=1)

    # Sum the right choice values, sum_right
    task_data['sum_right'] = task_data.iloc[:, 9:15].sum(axis=1)

    # Subtract sum_right from sum_left, delta_V
    task_data['delta_V'] = task_data['sum_left'] - task_data['sum_right']

    # Prepare data for Figure 4 Left
    # Create columns for wins in each frame, frame_1_win
    frame = 1

    for i in range(29, 35):
        task_data.insert(i, 'frame_' + str(frame) + '_win', '')

        frame += 1

    # frame_j_win value is +1 if left wins, -1 if right wins, and 0 if a tie
    for j in range(1, 7):
        task_data.loc[
                task_data['Frame ' + str(j) + ' Left Value'] >
                task_data['Frame ' + str(j) + ' Right Value'],
                'frame_' + str(j) + '_win'] = 1

        task_data.loc[
                task_data['Frame ' + str(j) + ' Left Value'] <
                task_data['Frame ' + str(j) + ' Right Value'],
                'frame_' + str(j) + '_win'] = -1

        task_data.loc[
                task_data['Frame ' + str(j) + ' Left Value'] ==
                task_data['Frame ' + str(j) + ' Right Value'],
                'frame_' + str(j) + '_win'] = 0

    # Create a column for the sum of wins, win_sum
    task_data['win_sum'] = task_data.iloc[:, 29:35].sum(axis=1)

    # Prepare data for Figure 4 Right
    # Create columns for absolute difference in each frame, frame_1_abs_diff
    frame = 1

    for i in range(36, 42):
        task_data.insert(i, 'frame_' + str(frame) + '_abs_diff', '')

        frame += 1

    # Calculate the absolute difference in each frame
    for j in range(1, 7):
        task_data['frame_' + str(j) + '_abs_diff'] = abs(
                task_data['Frame ' + str(j) + ' Left Value'] -
                task_data['Frame ' + str(j) + ' Right Value'])

    # Create a column for the sum of absolute differences, abs_diff_sum
    task_data['abs_diff_sum'] = task_data.iloc[:, 36:42].sum(axis=1)

    # |DATA GRAPHED IN FIGURE 3| #
    # Create bins for delta_V data
    # Number of equal bins, num_bins
    num_bins = 12

    # The left and right bin edges, left_edge, right_edge
    left_edge, right_edge = -110, 110

    bins = np.linspace(start=left_edge,
                       stop=right_edge,
                       num=num_bins,
                       endpoint=True)

    # Create bins for delta_V data, delta_V_bins
    task_data['delta_V_bins'] = pd.cut(task_data['delta_V'], bins=bins)

    # Create a DataFrame for binned data, binned_data
    binned_data = pd.DataFrame()

    # Create a column with the number of left choices, left_choice
    binned_data['left_choice'] = task_data['Left Chosen'].groupby(
        task_data['delta_V_bins']).sum()

    # Create a column with the number of values in each bin, val_in_bin
    binned_data['val_in_bin'] = task_data['delta_V_bins'].value_counts()

    # Create a column for the probability of left choice, prob_left
    # This is left_choice/val_in_bin
    binned_data['prob_left'] = binned_data[
            'left_choice']/binned_data['val_in_bin']

    # Create a column for the mean of delta_V in each bin, mean_delta_V
    binned_data['mean_delta_V'] = task_data.groupby(task_data['delta_V_bins']
                                                    )['delta_V'].mean()

    # Create a column for the standard deviation of delta_V in each bin,
    # std_delta_V
    binned_data['std_delta_V'] = task_data.groupby(
        task_data['delta_V_bins'])['delta_V'].std()

    # Create a column for the standard error of delta_V in each bin
    # sqrt(p.est*(1-p.est)/n), sd_err_delta_V
    binned_data['sd_err_delta_V'] = np.sqrt((
            binned_data['prob_left'] *
            (1 - binned_data['prob_left'])) /
            binned_data['val_in_bin'])

    # Create a column for the bins and sort the bins in ascending order
    binned_data = binned_data.sort_index().reset_index()

    print(binned_data)

    if graph_1:
        # Plot delta and the probabilities
        x_label = 'ΔV'
        y_label = 'Pr(Choose Left)'

        xtick_val = [-100, -40, 0, 40, 100]
        ytick_val = [0, 0.5, 1]

        # Set plot lables and show the plot
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.xticks(xtick_val)
        plt.yticks(ytick_val)
        plt.xlim(-100, 100)
        plt.ylim(0, 1)
        plt.grid(True)
        plt.plot(binned_data['mean_delta_V'], binned_data['prob_left'])
        plt.errorbar(binned_data['mean_delta_V'],
                     binned_data['prob_left'],
                     yerr=binned_data['sd_err_delta_V'],
                     fmt='o')

        plt.savefig(save_dest + 'binned_data_graph.png')

    # |END FIGURE 3| #

    binned_data.to_csv(save_dest + 'binned_data_table.csv')

    ################################################
    # |DATA GRAPHED IN FIGURE 4 LEFT| #
    # Create bins for wins data, win_bins

    task_data.loc[
            task_data['win_sum'] > 0,
            'win_bins'] = '> 0'

    task_data.loc[
            task_data['win_sum'] == 0,
            'win_bins'] = '0'

    task_data.loc[
            task_data['win_sum'] < 0,
            'win_bins'] = '< 0'

    # Create a DataFrame for binned data, binned_data
    binned_win_data = pd.DataFrame()

    # Create a column with the number of left choices, left_choice
    binned_win_data = pd.pivot_table(task_data,
                                     index=['win_bins', 'delta_V_bins'],
                                     values='Left Chosen',
                                     aggfunc=np.sum).rename(
                                             columns={'Left Chosen':
                                                      'left_choice'})

    # Create a column with the number of values in each bin, val_in_bin
    binned_win_data['val_in_bin'] = task_data.groupby(
            task_data['win_bins'])['delta_V_bins'].value_counts()

    # Create a column for the probability of left choice, prob_left
    # This is left_choice/val_in_bin
    binned_win_data['prob_left'] = binned_win_data['left_choice']\
                                              / binned_win_data['val_in_bin']

    # Create a column for the mean of delta_V in each bin, mean_delta_V
    binned_win_data['mean_delta_V'] = pd.pivot_table(task_data,
                                                     index=['win_bins',
                                                            'delta_V_bins'],
                                                     values='delta_V',
                                                     aggfunc=np.mean)

    # Create a column for the standard deviation of delta_V in each bin,
    # std_delta_V
    binned_win_data['std_delta_V'] = pd.pivot_table(task_data,
                                                    index=['win_bins',
                                                           'delta_V_bins'],
                                                    values='delta_V',
                                                    aggfunc=np.std)

    # Create a column for the standard error of delta_V in each bin
    # sqrt(p.est*(1-p.est)/n), sd_err_delta_V
    binned_win_data['sd_err_delta_V'] = np.sqrt((
            binned_win_data['prob_left'] *
            (1 - binned_win_data['prob_left'])) /
            binned_win_data['val_in_bin'])

    # Reset the DataFrame index
    binned_win_data = binned_win_data.reset_index()

    # Remove data for win_bins = 0
    binned_win_data = binned_win_data[binned_win_data['win_bins'] != '0']

    print(binned_win_data)

    if graph_2:
        # Plot delta and the probabilities
        x_label = 'ΔV'
        y_label = 'Pr(Choose Left)'

        xtick_val = [-60, -30, 0, 30, 60]
        ytick_val = [0, 0.5, 1]

        # Set plot lables and show the plot
        fig, ax = plt.subplots()
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.xticks(xtick_val)
        plt.yticks(ytick_val)
        plt.xlim(-100, 100)
        plt.ylim(0, 1)
        plt.grid(True)

        # Graph the subplots, excluding the index of 0
        for key, grp in binned_win_data.groupby(['win_bins']):
            if key != '0':
                ax = grp.plot(ax=ax,
                              kind='line',
                              x='mean_delta_V',
                              y='prob_left',
                              label=key)

                plt.errorbar(binned_win_data['mean_delta_V'],
                             binned_win_data['prob_left'],
                             yerr=binned_win_data['sd_err_delta_V'],
                             fmt='o')

        plt.savefig(save_dest + 'binned_win_graph.png')

    # |END FIGURE 4 LEFT| #

    binned_win_data.to_csv('binned_win_data_table.csv')

    # |DATA GRAPHED IN FIGURE 4 RIGHT| #
    # Create bins for absolute data, abs_diff_bins
    task_data.loc[
            task_data['abs_diff_sum'] > 90,
            'abs_diff_bins'] = '> 90'

    task_data.loc[
            task_data['abs_diff_sum'] <= 90,
            'abs_diff_bins'] = '<= 90'

    # Create a DataFrame for binned data, binned_abs_diff_data
    binned_abs_diff_data = pd.DataFrame()

    # Create a column with the number of left choices, left_choice
    binned_abs_diff_data = pd.pivot_table(task_data,
                                  index=['abs_diff_bins', 'delta_V_bins'],
                                  values='Left Chosen',
                                  aggfunc=np.sum).rename(
                                  columns={'Left Chosen': 'left_choice'})

    # Create a column with the number of values in each bin, val_in_bin
    binned_abs_diff_data['val_in_bin'] = task_data.groupby(
            task_data['abs_diff_bins'])['delta_V_bins'].value_counts()

    # Create a column for the probability of left choice, prob_left
    # This is left_choice/val_in_bin
    binned_abs_diff_data['prob_left'] = binned_abs_diff_data['left_choice']\
        / binned_abs_diff_data['val_in_bin']

    # Create a column for the mean of delta_V in each bin, mean_delta_V
    binned_abs_diff_data['mean_delta_V'] = pd.pivot_table(task_data,
                        index=['abs_diff_bins',
                               'delta_V_bins'],
                               values='delta_V',
                               aggfunc=np.mean)

    # Create a column for the standard deviation of delta_V in each bin,
    # std_delta_V
    binned_abs_diff_data['std_delta_V'] = pd.pivot_table(task_data,
                        index=['abs_diff_bins',
                               'delta_V_bins'],
                               values='delta_V',
                               aggfunc=np.std)

    # Create a column for the standard error of delta_V in each bin
    # sqrt(p.est*(1-p.est)/n), sd_err_delta_V
    binned_abs_diff_data['sd_err_delta_V'] = np.sqrt((
            binned_abs_diff_data['prob_left'] *
            (1 - binned_abs_diff_data['prob_left'])) /
            binned_abs_diff_data['val_in_bin'])

    print(binned_abs_diff_data)

    if graph_3:
        # Plot delta and the probabilities
        x_label = 'ΔV'
        y_label = 'Pr(Choose Left)'

        xtick_val = [-80, -40, 0, 40, 80]
        ytick_val = [0, 0.5, 1]

        # Set plot lables and show the plot
        fig, ax = plt.subplots()
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.xticks(xtick_val)
        plt.yticks(ytick_val)
        plt.xlim(-100, 100)
        plt.ylim(0, 1)
        plt.grid(True)

        # Graph the subplots, excluding the index of 0
        for key, grp in binned_abs_diff_data.groupby(['abs_diff_bins']):
            if key != '0':
                ax = grp.plot(ax=ax,
                              kind='line',
                              x='mean_delta_V',
                              y='prob_left',
                              label=key)

        plt.errorbar(binned_abs_diff_data['mean_delta_V'],
                     binned_abs_diff_data['prob_left'],
                     yerr=binned_abs_diff_data['sd_err_delta_V'],
                     fmt='o')

        plt.savefig(save_dest + 'abs_diff_graph.png')

    # |END FIGURE 4 RIGHT| #

    binned_abs_diff_data.to_csv(save_dest+'abs_diff_table.csv')

    task_data.to_csv(save_dest + 'task_data.csv')


task_data_graphs(task_file, save_dest)
