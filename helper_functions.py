import os

# needed for easier access to data without using absolute path
def get_data_path(filename):
    return os.path.join("Data", filename)

def get_data_path_root(filename):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # full path to /Tests
    project_root = os.path.abspath(os.path.join(current_dir, ".."))  # now correctly resolves to ML_Ops_T1
    return os.path.join(project_root, "ML_OPS_T1\\Data", filename)
