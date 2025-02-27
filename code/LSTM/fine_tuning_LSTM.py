import torch
import argparse

from LSTM_kuroda import LSTMClassification
from train_LSTM import train
from get_dataset import googledrive_download, init_dataset

# GPUチェック
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model')
    parser.add_argument('--fine_tuned_model')
    return parser.parse_args()


def main():

    args = parse_arguments()

    pretrained_model_path = args.pretrained_model
    fine_tuned_model_path = args.fine_tuned_model

    # the number of players
    num_player = 22

    batch_size = 2048
    input_dim = (num_player + 1) * 2
    hidden_dim = 20
    target_size = 18
    num_epochs = 1000
    lr = 0.01

    # 各戦術的行動の名前 
    tactical_action_name_list = ['Build up 1', 'Progression 1', 'Final third 1', 'Counter-attack 1', 'High press 1', 'Mid block 1', 'Low block 1', 'Counter-press 1', 'Recovery 1', 'Build up 2', 'Progression 2', 'Final third 2', 'Counter-attack 2', 'High press 2', 'Mid block 2', 'Low block 2', 'Counter-press 2', 'Recovery 2']

    # numpy load
    sequence_np, label_np = googledrive_download(bepro=True) # _0_or_1, 
    print(sequence_np)
    print(sequence_np.shape, label_np.shape)

    label_np = label_np # [:, 1:]

    train_loader, val_loader, test_loader = init_dataset(sequence_np, label_np, batch_size)

    # モデルを定義
    model = LSTMClassification(input_dim=input_dim, hidden_dim=hidden_dim, target_size=target_size)
    
    # 学習済みモデルのロード
    model.load_state_dict(torch.load(pretrained_model_path), strict=False)

    # 特定の層のみファインチューニング
    for param in model.parameters():
        param.requires_grad = False  # すべてのパラメータを固定
    for param in model.fc.parameters():
        param.requires_grad = True  # 最終分類層のみ更新

    model = train(train_loader, val_loader, test_loader, model, num_epochs, lr)
    torch.save(model.state_dict(), fine_tuned_model_path)


if __name__ == "__main__":
    main()