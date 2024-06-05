import tisc
import torch


from torch.utils.data import DataLoader, TensorDataset, random_split


def test():

    ############################ 各自のデータセットを読み込む ##########################

    # クラス情報の設定    
    # 各自のデータセットに合わせてクラスの情報を設定してください。

    # クラス数の設定 (int型)
    num_classes = "your class number"

    # クラスラベルの設定 (str型。e.g. ["はたきこみ", "押し出し", "突き落とし", "寄り切り"])
    class_labels = "your class labels"

    # 各自で作成した時系列データの、Test用データセットを読み込んでください。

    # データセットの読み込み（適宜書き換えてください。型はnumpy.ndarrayを想定しています。）
    test_X = "your test data"

    # ラベルの読み込み（適宜書き換えてください。型はnumpy.ndarrayを想定しています。）
    test_Y = "your test label"
    
    ################################################################################

    # PyTorchのデバイスを取得
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # データセットをテンソルに変換
    tensor_test_X = torch.tensor(test_X).float().to(device)
    tensor_test_Y = torch.tensor(test_Y).long().to(device)

    # データセットをTensorDatasetに変換
    test_dataset = TensorDataset(tensor_test_X, tensor_test_Y)

    # DataLoaderを作成
    batch_size = 2048
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # データの形状を取得
    test_iter = iter(test_loader)
    first_batch = next(test_iter)
    _, timestep, dimentions = first_batch[0].shape

    # モデルの構築
    classifier = tisc.build_classifier(model_name="LSTM",
                                       timestep=timestep,
                                       dimentions=dimentions,
                                       num_classes=num_classes,
                                       class_labels=class_labels)
    
    # モデルの読み込み
    # 学習後に保存されたモデルのパスをmodel_pathに指定してください。
    model_path = "your model path"
    classifier.load_model(model_path=model_path)

    # モデルの評価
    classifier.evaluate(test_loader,
                        return_report=True,
                        return_confusion_matrix=True,
                        with_best_model=True)
    
    # モデルの推論
    # 返り値は(predicted_label, outputs)のタプルです。

    # まとめて推論（データセット単位など）
    print(classifier.predict(first_batch[0]))

    # 1データずつ推論
    print(classifier.predict(first_batch[0][0]))


if __name__ == "__main__":
    test()