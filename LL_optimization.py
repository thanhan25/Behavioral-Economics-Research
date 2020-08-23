'''
This function uses experimental data to find an optimized Log Likelihiood.
'''

# Import lmfit, numpy, pandas, and pyplot
from lmfit import Model
import numpy as np
import pandas as pd


# Create a function for the psychometric curve, psychometric_function
def psychometric_function(x, alpha):

    return (np.exp(x * alpha))/((np.exp(x * alpha)) + (np.exp(x * alpha * -1)))


# Prepare the spreadsheet data

# Assign spreadsheet file name, task_file
task_file = 'PriceObfuscation_Meeting_2019September25/table_task1.csv'

# Assign save destination
save_dest = 'Task 1/'


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

# Sum the left choice values, sum_left
task_data['sum_left'] = task_data.iloc[:, 3:9].sum(axis=1)

# Sum the right choice values, sum_right
task_data['sum_right'] = task_data.iloc[:, 9:15].sum(axis=1)

# Subtract sum_right from sum_left, delta_V
task_data['delta_V'] = task_data['sum_left'] - task_data['sum_right']


# Use lmfit Model to evaluate the optimized alpha value
# Model determines the parameter and independent variabled
gmodel = Model(psychometric_function)
print('parameter names: {}'.format(gmodel.param_names))
print('independent variables: {}'.format(gmodel.independent_vars))

# Set the initial values for the model parameters
params = gmodel.make_params(alpha=0.2)

y = psychometric_function(task_data['delta_V'], 0.2)

result = gmodel.fit(y,
                    params,
                    x=task_data['delta_V'])

print(result.fit_report())
