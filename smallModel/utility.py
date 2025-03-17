import pandas as pd


def compare_csv(csv1, csv2, output_name):

    df1 = pd.read_csv(csv1).set_index('Metric')
    df2 = pd.read_csv(csv2).set_index('Metric')

    diff = df1.subtract(df2)
    comp = df1.compare(diff).round(4)


    with open(f'smallModel/outputs/{output_name}.csv', "w") as file:
        file.write(comp.to_csv())

def side_by_side(csv1, csv2, output_name):

    df1 = pd.read_csv(csv1).set_index('Metric')
    df2 = pd.read_csv(csv2).set_index('Metric')

    comp = df1.compare(df2).round(4)


    with open(f'smallModel/outputs/{output_name}.csv', "w") as file:
        file.write(comp.to_csv())

path1 = 'smallModel/outputs/crossval_test_metrics.csv'
path2 = 'smallModel/outputs/crossval_ctrl_metrics.csv'
side_by_side(path1, path2, 'crossval_comparison')