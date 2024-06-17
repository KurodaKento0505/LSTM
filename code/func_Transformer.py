import tisc
import torch
import numpy as np

from torch.utils.data import DataLoader, TensorDataset, random_split

'''GPUチェック'''
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


def Transformer(sequence_np, label_np, num_tactical_action_per_training, tactical_action_name, dim_of_image, train, make_graph, val):


    ##################################################################
    batch_size = 512
    hidden_dim = 20
    epoch = 500
    lr = 0.01
    ##################################################################



    # slice する数を計算
    slice_len = len(sequence_np) % 10
    sequence_np = np.delete(sequence_np, slice(len(sequence_np) - slice_len, len(sequence_np)), 0)
    label_np = np.delete(label_np, slice(len(sequence_np) - slice_len, len(sequence_np)), 0)


    # torch.tensorでtensor型に
    train_x = torch.from_numpy(sequence_np.astype(np.float32)).clone()
    train_t = torch.from_numpy(label_np.astype(np.float32)).clone()

    print('train_x:', train_x.shape)
    print('train_t:', train_t.shape)


    if train:
        dataset = torch.utils.data.TensorDataset(train_x, train_t)

        train_size = int(len(dataset) * 0.8) # train_size is 3000
        val_size = int(len(dataset) * 0.1) # val_size is 1000
        test_size = int(len(dataset) * 0.1)# val_size is 1000
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size], torch.Generator().manual_seed(3)) # 42
        
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle = True, num_workers = 0)
        valloader = torch.utils.data.DataLoader(val_dataset, batch_size, shuffle = True, num_workers = 0)
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle = True, num_workers = 0)


        # データの形状を取得
        train_iter = iter(trainloader)
        first_batch = next(train_iter)
        _, timestep, dimentions = first_batch[0].shape

        # モデルの構築
        classifier = tisc.build_classifier(model_name="Transformer",
                                        timestep=timestep,
                                        dimentions=dimentions,
                                        num_classes=num_tactical_action_per_training,
                                        tactical_action_name = tactical_action_name,
                                        train_truth = True)

        # モデルの学習
        # epochsは150に設定していますが、各自で調整してください。
        # 学習率はデフォルトでは0.001に設定されていますが、任意の値を設定する場合は、引数lrで指定してください。
        # 学習後のモデルは"./tisc_output/(モデル名)/(実行した時間)/weights"ディレクトリに保存されます。
        classifier.train(epochs=epoch,
                        train_loader=trainloader,
                        tactical_action_name = tactical_action_name,
                        val_loader=valloader,
                        lr = lr)


    else:
        if make_graph:
            graph_test_dataset = torch.utils.data.TensorDataset(train_x, train_t)
            graph_testloader = torch.utils.data.DataLoader(graph_test_dataset, batch_size, shuffle = False, num_workers = 0)


            # データの形状を取得
            graph_test_iter = iter(graph_testloader)
            first_batch = next(graph_test_iter)
            _, timestep, dimentions = first_batch[0].shape

            # モデルの構築
            classifier = tisc.build_classifier(model_name="Transformer",
                                            timestep=timestep,
                                            dimentions=dimentions,
                                            num_classes=num_tactical_action_per_training,
                                            tactical_action_name = tactical_action_name,
                                            train_truth = True)


            outputs_list, labels_list, len_loader = classifier.evaluate(graph_testloader)

            return outputs_list, labels_list, len_loader

        else:
            dataset = torch.utils.data.TensorDataset(train_x, train_t)
            train_size = int(len(dataset) * 0.6) # train_size is 3000
            val_size = int(len(dataset) * 0.2) # val_size is 1000
            test_size = int(len(dataset) * 0.2)# val_size is 1000
            train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size], torch.Generator().manual_seed(3)) # 42
            trainloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle = True, num_workers = 0)
            valloader = torch.utils.data.DataLoader(val_dataset, batch_size, shuffle = True, num_workers = 0)
            testloader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle = True, num_workers = 0)


if __name__ == "__main__":
    Transformer()