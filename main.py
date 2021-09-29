from data_preprocessing import load_and_split_data

if __name__ == '__main__':
    X, y = load_and_split_data()
    print(y)
