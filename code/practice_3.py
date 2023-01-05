from pratice_2023_fdu.code.practice_2 import read_orl_image, split_train_test_orl

if __name__ == '__main__':
    image_file_path = "../data/ORL"
    out_file_path = "../data/ORL/"
    out_res_path = "../results/"
    fea, label = read_orl_image(image_file_path, out_file_path)
    train_x, train_y, test_x, test_y = split_train_test_orl(fea, label)