import argparse
import os
import imageio
from matplotlib import pyplot as plt
import numpy as np
import utils
from definitions import OUT_DIR, device
import torch
from dataloader import get_dataloader, get_dataset
import copy
from train_cnn import reconstruct_h2, Net1, Net3, reconstruct_h1

def main(model_label, steps=35, sig_factor=3):
    model_dir = OUT_DIR.joinpath(model_label)
    sd_mdl = torch.load(str(model_dir.joinpath("model.pt")), map_location=torch.device('cpu'))
    args = sd_mdl["args"]
    H1, H2, L1, L2, W1, W2 = sd_mdl["H1"].to(device), sd_mdl["H2"].to(device), sd_mdl["L1"].to(device), sd_mdl["L2"].to(device), sd_mdl["W1"].to(device), sd_mdl["W2"].to(device)

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load Training Data
    def get_data(d):
        loader = get_dataloader(argparse.Namespace(**d))
        x, y = get_dataset(loader)
        return x, y
    args_dataset = copy.copy(args.dataset)
    x_train, y_train = get_data(utils.merge_two_dicts(args_dataset, {"train": True}))
    img_shape = list(x_train.shape[1:])

    nChannels = x_train.shape[1]
    if "phi1" in sd_mdl:
        cnn_kwargs = dict(kernel_size=4, stride=2, padding=1)
        if np.prod(x_train.shape[1:]) <= 28 * 28 * 3:
            cnn_kwargs = (cnn_kwargs, dict(kernel_size=3, stride=1), 5)
        else:
            cnn_kwargs = cnn_kwargs, cnn_kwargs, 8
        phi1 = Net1(nChannels, capacity=args.levels.j1.phi.capacity, x_fdim1=args.levels.j1.phi.x_fdim1, x_fdim2=args.levels.j1.phi.x_fdim2, cnn_kwargs=cnn_kwargs).to(device)
        psi1 = Net3(nChannels, capacity=args.levels.j1.phi.capacity, x_fdim1=args.levels.j1.phi.x_fdim1, x_fdim2=args.levels.j1.phi.x_fdim2, cnn_kwargs=cnn_kwargs).to(device)
        phi1.load_state_dict(sd_mdl["phi1"])
        psi1.load_state_dict(sd_mdl["psi1"])

    # Save Images in the directory
    if not os.path.exists('Traversal_imgs/{}'.format(model_label)):
        os.makedirs('Traversal_imgs/{}'.format(model_label))

    # Reconstruction and Generation
    with torch.no_grad():
        # Visualize quality of reconstructed samples
        perm1 = torch.randperm(H1.shape[0])
        m = 10
        fig2, ax = plt.subplots(1, m*m)
        it = 0
        for i in range(m):
            for j in range(m):
                ax[i*m+j].imshow(((x_train[perm1[it]].cpu().permute(1,2,0))), cmap='gray')
                it += 1
                imageio.imwrite(
                    'Traversal_imgs/{}/orig_{}.png'.format(model_label, it),
                    x_train[perm1[it]].cpu().permute(1, 2, 0))
        plt.suptitle('Ground Truth')
        plt.setp(ax, xticks=[], yticks=[])
        plt.savefig("truth.png")

        for level in range(1):
            for component in range(1):
                fig1, ax = plt.subplots(1, m*m)
                x_train_hat = reconstruct_h2(W1, W2, H2, psi1)
                it = 0
                for i in range(m):
                    for j in range(m):
                        ax[i*m+j].imshow(x_train_hat[perm1[it]].cpu().permute(1,2,0), cmap='gray')
                        it += 1
                        imageio.imwrite(
                            'Traversal_imgs/{}/rec_{}.png'.format(model_label, it),
                            x_train_hat[perm1[it]].cpu().permute(1, 2, 0))
                plt.suptitle(f'Reconstructed training samples')
                plt.setp(ax, xticks=[], yticks=[])
                plt.savefig("rec.png")

        # Traversals along deep principal components
        for k, name in enumerate(["h1", "h2"]):
            h = H2 if k == 1 else H1
            for i in range(h.shape[1]):
                dim = i
                m = steps  # Number of steps
                mul_off = 0.0  # (for no-offset, set multiplier to 0)

                # Set the principal component range from Gaussian
                mu, sig = torch.mean(h[:,i]), torch.std(h[:,i])
                lambd = torch.linspace(mu-sig_factor*sig, mu+sig_factor*sig, steps=m)

                uvec = torch.FloatTensor(torch.zeros(h.shape[1]))
                uvec[dim] = 1  # unit vector
                yoff = mul_off * torch.ones(h.shape[1]).float()
                yoff[dim] = 0

                yop = yoff.repeat(lambd.size(0), 1) + torch.mm(torch.diag(lambd),
                                                               uvec.repeat(lambd.size(0), 1))  # Traversal vectors
                yop = yop.to(device)

                if k == 1:
                    x_gen = reconstruct_h2(W1, W2, yop, psi1)
                elif k == 0:
                    x_gen = reconstruct_h1(W1, yop, psi1)

                for j in range(x_gen.shape[0]):
                    imageio.imwrite(
                        'Traversal_imgs/{}/{}_{}_{}.png'.format(model_label, name, dim, j),
                        x_gen[j].cpu().permute(1,2,0))

    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=str, default='3pp8rxvi', help='Enter model label')
    opt_gen = parser.parse_args()
    main(opt_gen.model)
