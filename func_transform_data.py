import numpy as np

def transform(sequence_np):

    # 変数宣言
    data_size = int(sequence_np.shape[0])
    seq_length = int(sequence_np.shape[1])
    input_size = int(sequence_np.shape[2]) * int(sequence_np.shape[3]) * int(sequence_np.shape[4])

    # 学習データ初期化
    train_x = np.zeros((data_size, seq_length, input_size))

    # 学習データ数分ループ
    for i in range(data_size):
        # 1枚の画像に含まれるシーケンスデータの数分ループ
        for j in range(seq_length):
            # 行方向にループ
            for k in range(int(sequence_np.shape[2])):
                # 列方向にループ
                for l in range(int(sequence_np.shape[3])):
                    # channel方向にループ
                    for m in range(int(sequence_np.shape[4])):
                        # input_dataにtrain_imgsのn_time分のシーケンスデータを入れていく
                        train_x[i, j, k * int(sequence_np.shape[3]) + l * int(sequence_np.shape[4]) + m] = sequence_np[i, j, k, l, m]

        if i % 1000 == 0:
            print(i)

    return train_x


'''# 逆
# 学習データ初期化
input_data = np.zeros((data_size, seq_length, data_variable_list[2], data_variable_list[3]))

# 学習データ数分ループ
for i in range(data_size):
    # 1枚の画像に含まれるシーケンスデータの数分ループ
    for j in range(seq_length):
        # 行方向にループ
        for k in range(data_variable_list[2]):
            # 列方向にループ
            for l in range(data_variable_list[3]):
                # input_dataにtrain_imgsのn_time分のシーケンスデータを入れていく
                input_data[i, j, k, l] = train_x[i, j, k * data_variable_list[3] + l]'''