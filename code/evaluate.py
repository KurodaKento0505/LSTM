import torch
import pandas as pd
import numpy as np
import argparse
from LSTM.LSTM_kuroda import LSTMClassification
from get_dataset import googledrive_download, init_dataset

# GPUチェック
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    parser.add_argument('--output_file')
    parser.add_argument('--make_graph', action='store_true')
    return parser.parse_args()


def main():
    args = parse_arguments()
    model_path = args.model
    output_file = args.output_file
    make_graph = args.make_graph

    # the number of players
    num_player = 22

    batch_size = 2048
    input_dim = (num_player + 1) * 2
    hidden_dim = 20
    target_size = 18

    # 各戦術的行動の名前 
    tactical_action_name_list = ['Build up 1', 'Progression 1', 'Final third 1', 'Counter-attack 1', 'High press 1', 'Mid block 1', 'Low block 1', 'Counter-press 1', 'Recovery 1', 'Build up 2', 'Progression 2', 'Final third 2', 'Counter-attack 2', 'High press 2', 'Mid block 2', 'Low block 2', 'Counter-press 2', 'Recovery 2']

    # 学習済みモデルのロード
    model = LSTMClassification(input_dim=input_dim, hidden_dim=hidden_dim, target_size=target_size)
    model.load_state_dict(torch.load(model_path), strict=False)

    if make_graph:
        sequence_np, label_np = googledrive_download(make_graph=make_graph, bepro=True)
        print(sequence_np.shape, label_np.shape)
        graph_testloader = init_dataset(sequence_np, label_np, batch_size, make_graph=True)
        outputs_list, labels_list = evaluate(model, graph_testloader)
        outputs_np = np.array(outputs_list)
        labels_np = np.array(labels_list)
        print(labels_np.shape, outputs_np.shape)
        df = generate_sequence_result(outputs_np, labels_np, tactical_action_name_list)

    else:
        # numpy load
        sequence_np, label_np = googledrive_download(bepro=True) 
        print(sequence_np.shape, label_np.shape)
        train_loader, val_loader, test_loader = init_dataset(sequence_np, label_np, batch_size)
        outputs_list, labels_list = evaluate(model, test_loader)
        outputs_np = np.array(outputs_list)
        labels_np = np.array(labels_list)
        print(labels_np.shape, outputs_np.shape)
        df = get_error(outputs_np, labels_np, tactical_action_name_list)

    # CSVファイルに保存
    df.to_csv(output_file, index=False)
    print(f"Output file saved to {output_file}")


def evaluate(model, loader):
    model = model.to(device)
    model.eval()

    outputs_list = []
    labels_list = []

    with torch.no_grad():
        for i, data in enumerate(loader):
            inputs, labels_i = data
            inputs,labels_i = inputs.to(device), labels_i.to(device)
            labels_i_list = labels_i.tolist()

            outputs_i = model(inputs)
            # outputs_i = F.softmax(outputs_i, dim = 1)
            outputs_i_list = outputs_i.tolist()

            for j in range(len(outputs_i)):
                outputs_list.append(outputs_i_list[j])

            for j in range(len(labels_i)):
                labels_list.append(labels_i_list[j])

    return outputs_list, labels_list


# modelの評価
def get_error(outputs_np, labels_np, tactical_action_name_list, half=1):
    # test_dataによる精度評価
    # 各列の誤差を計算
    errors_np = np.abs(outputs_np - labels_np)  # 絶対誤差
    mean_errors = np.mean(errors_np, axis=0)    # 各列の平均誤差
    std_errors = np.std(errors_np, axis=0)      # 各列の誤差の標準偏差
    
    # 全体の誤差を計算
    overall_mean_error = np.mean(errors_np)  # 全体の平均誤差
    overall_std_error = np.std(errors_np)    # 全体の誤差の標準偏差

    # 結果を出力
    # データフレームを作成
    data = {
        "Tactical Action": tactical_action_name_list,
        "Mean Error": mean_errors,
        "Standard Deviation of Error": std_errors
    }
    error_df = pd.DataFrame(data)
    
    # 全体の誤差を出力
    overall_error_data = pd.DataFrame({
        "Tactical Action": ["Overall"],
        "Mean Error": [overall_mean_error],
        "Standard Deviation of Error": [overall_std_error]
    })
    
    return pd.concat([error_df, overall_error_data], ignore_index=True)


def generate_sequence_result(outputs_np, labels_np, tactical_action_name_list, half=1):

    # 新しいdf
    sequence_outcome_df = pd.DataFrame()

    slice_len = len(labels_np) % len(outputs_np)
    labels_np = np.delete(labels_np, slice(len(labels_np) - slice_len, len(labels_np)), 0)

    for j in range(len(tactical_action_name_list)):
        sequence_outcome_df[tactical_action_name_list[j]] = labels_np[:, j]
        sequence_outcome_df['output_' + tactical_action_name_list[j]] = outputs_np[:, j]

    return sequence_outcome_df


if __name__ == '__main__':
    main()