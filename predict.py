
import torch
import torch.nn as nn
import numpy as np


def predict(model, model_dir, results_dir, features_path, vid_list_file, epoch, actions_dict, device, sample_rate, args):

    # collect arguments
    verbose = args.verbose
    use_best_model = args.use_best_model

    # multi-GPU
    if args.multi_gpu and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.eval()

    with torch.no_grad():
        model.to(device)
        if use_best_model == 'source':
            model.load_state_dict(torch.load(model_dir + "/acc_best_source.model"))
        elif use_best_model == 'target':
            model.load_state_dict(torch.load(model_dir + "/acc_best_target.model"))
        else:
            model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model"))
        file_ptr = open(vid_list_file, 'r')
        list_of_vids = file_ptr.read().split('\n')[:-1]  # testing list
        file_ptr.close()
        for vid in list_of_vids:
            if verbose:
                print(vid)  

            features = np.load(features_path + vid.split('.')[0] + '.npy')
            features = features[:, ::sample_rate]
            input_x = torch.tensor(features, dtype=torch.float)
            input_x.unsqueeze_(0)
            input_x = input_x.to(device)
            mask = torch.ones_like(input_x)
            predictions, _, _, _, _, _, _, _, _, _, _, _, _, _ = model(input_x, input_x, mask, mask, [0, 0], reverse=False)
            _, predicted = torch.max(predictions[:, -1, :, :].data, 1)
            predicted = predicted.squeeze()
            recognition = []
            for i in range(predicted.size(0)):
                recognition = np.concatenate((recognition,
                    [list(actions_dict.keys())[list(actions_dict.values()).index(predicted[i].item())]] * sample_rate))
            f_name = vid.split('/')[-1].split('.')[0]
            f_ptr = open(results_dir + "/" + f_name, "w")
            f_ptr.write("### Frame level recognition: ###\n")
            f_ptr.write(' '.join(recognition))
            f_ptr.close()
