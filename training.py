import tisc
import torch


from torch.utils.data import DataLoader, TensorDataset, random_split


def training():

    ############################ 各自のデータセットを読み込む ##########################

    # クラス情報の設定    
    # 各自のデータセットに合わせてクラスの情報を設定してください。

    # クラス数の設定 (int型)
    num_classes = "your class number"

    # クラスラベルの設定 (str型。e.g. ["はたきこみ", "押し出し", "突き落とし", "寄り切り"])
    class_labels = "your class labels"

    # 各自で作成した時系列データの、Training用データセットを読み込んでください。
    # データセットの形状は、（データ数，時系列長，特徴量の次元）となります。
    # またデータセットのラベルは、（データ数，）となります。

    # データセットの読み込み（適宜書き換えてください。型はnumpy.ndarrayを想定しています。）
    train_X = "your data"

    # ラベルの読み込み（適宜書き換えてください。型はnumpy.ndarrayを想定しています。）
    train_Y = "your label"
    
    ################################################################################

    # PyTorchのデバイスを取得
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # データセットをテンソルに変換
    tensor_train_X = torch.tensor(train_X).float().to(device)
    tensor_train_Y = torch.tensor(train_Y).long().to(device)

    # データセットをTensorDatasetに変換
    dataset = TensorDataset(tensor_train_X, tensor_train_Y)

    # データセットを訓練データと検証データに分割
    # ここでは、訓練データと検証データの割合を8:2に設定しています。
    # 分割の割合は、各自で調整してください。
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # DataLoaderを作成
    # batch_sizeは2048に設定していますが、GPUメモリ等に合わせて各自で調整してください。
    batch_size = 2048
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # データの形状を取得
    train_iter = iter(train_loader)
    first_batch = next(train_iter)
    _, timestep, dimentions = first_batch[0].shape

    # モデルの構築
    classifier = tisc.build_classifier(model_name="LSTM",
                                    timestep=timestep,
                                    dimentions=dimentions,
                                    num_classes=num_classes,
                                    class_labels=class_labels)
    
    # モデルの学習
    # epochsは150に設定していますが、各自で調整してください。
    # 学習率はデフォルトでは0.001に設定されていますが、任意の値を設定する場合は、引数lrで指定してください。
    # 学習後のモデルは"./tisc_output/(モデル名)/(実行した時間)/weights"ディレクトリに保存されます。
    classifier.train(epochs=150,
                    train_loader=train_loader,
                    val_loader=val_loader)


if __name__ == "__main__":
    training()