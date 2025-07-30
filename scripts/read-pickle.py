import pickle
import sys

if len(sys.argv) < 2:
    print("Usage: python read_pickle.py <pickle_file>")
    sys.exit(1)

pickle_file_path = sys.argv[1]

try:
    with open(pickle_file_path, 'rb') as f:
        data = pickle.load(f)
    for line in data:
        print(line)
        print('------------------------------------------')
    #print(data) # This will print the loaded Python object
except FileNotFoundError:
    print(f"Error: File '{pickle_file_path}' not found.")
    sys.exit(1)
except pickle.UnpicklingError:
    print(f"Error: Could not unpickle '{pickle_file_path}'. It might be corrupted or not a valid pickle file.")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    sys.exit(1)