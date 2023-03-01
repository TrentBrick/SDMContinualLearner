import torchvision
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from sklearn.preprocessing import StandardScaler
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


def analyze_training(
    testloader,
    net,
    params,
    additional_embeddings=True,
    FFN=False,
    save_outputs=False,
    dirr=None,
):

    all_outputs, all_labels, first_iter = [], [], True
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            if FFN:
                outputs, meta = net(images.view(images.shape[0], -1))
                num_actives, active_values, cosine_sims = meta

            else:
                outputs, meta = net(images.view(images.shape[0], -1), params.epochs_to_train_for)
                (
                    inhibs,
                    cosine_sims,
                    num_actives,
                    active_values,
                    min_top_k_vals,
                    inhib_input_sum,
                    k_val_used,
                ) = meta
            all_outputs += list(outputs.detach().numpy())
            all_labels += list(labels)

            if first_iter:
                neuron_activations = active_values
                first_iter = False
            else:
                neuron_activations = np.concatenate(
                    [neuron_activations, active_values], axis=0
                )

    all_outputs = np.asarray(all_outputs)
    all_labels = np.asarray(all_labels)
    neuron_activations = np.asarray(neuron_activations)

    if save_outputs:
        save_str = dirr + "isFFN=" + str(FFN) + "k=" + str(params.k_min)
        suffix = ".png"
    else:
        save_str = None

    # Firing preferences for each neuron

    plt.figure()
    counts, o1, o2 = plt.hist(neuron_activations.sum(0), bins=200)
    plt.title(
        "Sum of activations across all "
        + str(len(neuron_activations))
        + " inputs for each neuron"
    )
    if save_outputs:
        plt.gcf().savefig(save_str + "SumActivations" + suffix, dpi=200)
    plt.show()

    """plt.figure()
    plt.xlim((0,50))
    counts, o1, o2 = plt.hist(neuron_activations.sum(0), bins=200)
    plt.title("Sum of activations across all "+str( len(neuron_activations) )+" inputs for each neuron")
    plt.show()"""

    if FFN:
        active_threshold = 120  # some neurons are barely active but de facto inactive!
    else:
        active_threshold = 3

    print(counts[:5], o1[:5])

    print(
        "fraction of neurons that are never activated:",
        counts[0] / neuron_activations.shape[1],
        "mask approach",
        (neuron_activations.sum(0) < active_threshold).sum()
        / neuron_activations.shape[1],
    )

    # find out what activates each neuron the most:
    df = pd.DataFrame(
        neuron_activations, columns=list(np.arange(params.nneurons[0]))
    )
    df["Labels"] = all_labels
    # maximum activation for each neuron!
    neuron_max_activations = df.groupby("Labels").sum().idxmax().values

    plt.figure()
    plt.hist(neuron_max_activations)
    ax = plt.gca()
    plt.title("Main Activating Label for each Neuron")
    ax.xaxis.set_major_locator(plt.MaxNLocator(10))
    if save_outputs:
        plt.gcf().savefig(save_str + "MainActivatingLabel" + suffix, dpi=200)
    plt.show()

    # Inter Digit Centroid Distances Between Digits
    if FFN:
        neuron_addresses = net.fc1.weight.detach().numpy()  # neurons x dendrites
        neuron_value_vs = net.fc2.weight.detach().numpy().T
    else:
        neuron_addresses = net.fc1.detach().numpy()  # neurons x dendrites
        neuron_value_vs = net.fc2.detach().numpy().T
    print(
        "Neuron addresses should be 1000x784!!!",
        neuron_addresses.shape,
        neuron_value_vs.shape,
    )
    centroids, dists = [], []
    for n in np.arange(10):
        match_inds = np.arange(neuron_addresses.shape[0])[n == neuron_max_activations]
        centroid = np.mean(neuron_addresses[match_inds], axis=0)
        centroids.append(centroid)
    centroids = np.asarray(centroids)
    for i in range(len(centroids)):
        row_dists = []
        for j in range(len(centroids)):
            eu_dist = np.sum((centroids[i] - centroids[j]) ** 2)
            cos_dist = (
                centroids[i].T
                @ centroids[j]
                / (np.linalg.norm(centroids[i]) * np.linalg.norm(centroids[j]))
            )
            row_dists.append((eu_dist, cos_dist))
        dists.append(row_dists)
    dists = np.asarray(dists)
    dists.shape

    print("Euclidian distance")
    display(pd.DataFrame(dists[:, :, 0]))

    print("cosine similarity")
    display(pd.DataFrame(dists[:, :, 1]))

    # Embeddings of the addresses and their receptive fields.
    scaled_neuron_addresses = StandardScaler().fit_transform(neuron_addresses)
    embed = pca_umap_analysis(
        scaled_neuron_addresses, neuron_max_activations, "Neuron Addresses"
    )
    fancy_umap_embed(
        neuron_addresses,
        neuron_max_activations,
        embed,
        params.img_dim,
        save_str=save_str,
    )

    # samples from all neurons. not just active ones.
    nsamps = 60
    rand_inds = np.random.choice(np.arange(0, neuron_addresses.shape[0]), nsamps)
    rand_addresses = neuron_addresses[rand_inds, :].reshape(
        (-1, 1, params.img_dim, params.img_dim)
    )

    print("Totally random, not most active.")
    plt.figure(figsize=(20, 12))
    grid = torchvision.utils.make_grid(
        torch.Tensor(rand_addresses), padding=2, normalize=True, nrow=10
    )
    # show images
    imshow(grid, str(nsamps) + " random neuron receptive fields")

    if not params.classification:
        rand_value_vs = neuron_value_vs[rand_inds, :].reshape(
            (-1, 1, params.img_dim, params.img_dim)
        )
        plt.figure(figsize=(20, 12))
        grid = torchvision.utils.make_grid(
            torch.Tensor(rand_value_vs), padding=2, normalize=True, nrow=10
        )
        # show images
        imshow(grid, str(nsamps) + " random neuron value vectors")

    # Same as above but for only those neurons that are active.
    active_mask = neuron_activations.sum(0) > active_threshold

    embed = pca_umap_analysis(
        scaled_neuron_addresses[active_mask],
        neuron_max_activations[active_mask],
        "Only Active Ones! " + str(active_mask.sum()) + "Neuron Addresses",
    )
    fancy_umap_embed(
        neuron_addresses[active_mask],
        neuron_max_activations[active_mask],
        embed,
        params.img_dim,
        save_str=save_str,
        most_active=True,
    )

    print("Giving some random plots from the most active neurons")
    nsamps = 60
    rand_inds = np.random.choice(np.arange(len(active_mask))[active_mask], nsamps)
    rand_addresses = neuron_addresses[rand_inds, :].reshape(
        (-1, 1, params.img_dim, params.img_dim)
    )
    plt.figure(figsize=(20, 12))
    grid = torchvision.utils.make_grid(
        torch.Tensor(rand_addresses), padding=2, normalize=True, nrow=10
    )
    # show images
    imshow(grid, str(nsamps) + " random most active neuron receptive fields")

    if not params.classification:
        rand_value_vs = neuron_value_vs[rand_inds, :].reshape(
            (-1, 1, params.img_dim, params.img_dim)
        )
        plt.figure(figsize=(20, 12))
        grid = torchvision.utils.make_grid(
            torch.Tensor(rand_value_vs), padding=2, normalize=True, nrow=10
        )
        # show images
        imshow(grid, str(nsamps) + " random most active neuron value vectors")

    # plot 5 of them. with their activation values.
    for i in range(5):
        r_ind = rand_inds[i]
        address_title = (
            "Most Activating Digit: "
            + str(neuron_max_activations[r_ind])
            + " \nNeuron Address for Very Active Ind: "
            + str(r_ind)
            + " \nSum of activations was: "
            + str(neuron_activations.sum(0)[r_ind])
        )

        if not params.classification:
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
            axes[0].imshow(
                neuron_addresses[r_ind, :].reshape((params.img_dim, params.img_dim))
            )
            axes[1].imshow(
                neuron_value_vs[r_ind, :].reshape((params.img_dim, params.img_dim))
            )
            axes[0].set_title(address_title)
            axes[1].set_title("Neuron Value Vector")
            fig.tight_layout()
            plt.show()

        else:

            plt.figure(figsize=(10, 10))
            plt.imshow(
                neuron_addresses[r_ind, :].reshape((params.img_dim, params.img_dim))
            )
            plt.title(address_title)
            plt.show()

        print("=========================")

    # Embeddings Predictions

    if additional_embeddings:
        all_outputs_scaled = StandardScaler().fit_transform(all_outputs)
        scaled_neuron_activations = StandardScaler().fit_transform(neuron_activations)

        pca_umap_analysis(all_outputs_scaled, all_labels, "Network Output Vector")
        pca_umap_analysis(
            scaled_neuron_activations, all_labels, "Hidden Layer Neurons Activated"
        )


# Plot a bunch of receptive fields
def imshow(img, title="No Title Given"):
    # img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.show()


def get_cosine_sims(images, net, k, norm):
    with torch.no_grad():
        x = images.view(images.shape[0], -1).T  # data x batch.
        # print(self.fc1.shape, x.shape)
        if norm:
            x = torch.matmul(
                net.fc1 / torch.norm(net.fc1, dim=1, keepdim=True),
                x / torch.norm(x, dim=0),
            )
        else:
            x = torch.matmul(net.fc1, x)
        cosine_sims = x.T

        vals, inds = torch.topk(cosine_sims, k, dim=1)
        lowest_cos_sims, _ = torch.min(vals, dim=1)

        # doing this so can be used for the reconstruction classifier
        top_k_mask = torch.zeros_like(cosine_sims)
        top_k_mask = top_k_mask.scatter(1, inds, 1)

        return cosine_sims, lowest_cos_sims, top_k_mask


def density_analysis(testloader, params, net):

    cosine_sims, lowest_cos_sims, all_images, input_img_ind = (
        [],
        [],
        np.zeros((10000, params.img_dim, params.img_dim)),
        0,
    )

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            t_cosine_sims, t_lowest_cos_sims, _ = get_cosine_sims(
                images, net, params.k_min, params.norm_addresses
            )

            cosine_sims += list(t_cosine_sims.detach().numpy())
            lowest_cos_sims += list(t_lowest_cos_sims.detach().numpy())
            all_images[input_img_ind : (input_img_ind + images.shape[0])] = (
                images.squeeze().detach().numpy()
            )
            input_img_ind += images.shape[0]

    ## Density using test images. Finding the furthest top-k neuron

    print("Min top k=", params.k_min)

    print(
        "lowest cosine sims for the top k values",
        len(cosine_sims),
        len(lowest_cos_sims),
        lowest_cos_sims[:10],
    )
    # print( lowest_cos_sims.shape, lowest_cos_sims.reshape(-1).shape)
    plt.hist(lowest_cos_sims, bins=50)
    abs_lowest_ind = np.argmin(lowest_cos_sims)  # for all examples
    abs_highest_ind = np.argmax(lowest_cos_sims)  # for all examples

    plt.imshow(all_images[abs_lowest_ind, :, :])
    plt.title(
        "Real Image - lowest topk cosine sim: " + str(lowest_cos_sims[abs_lowest_ind])
    )
    plt.show()
    plt.imshow(all_images[abs_highest_ind, :, :])
    plt.title(
        "Real Image - highest topk cosine sim: " + str(lowest_cos_sims[abs_highest_ind])
    )
    plt.show()

    ## Same with Random Patterns

    # doing the same as above but with random noise!

    rand = torch.Tensor(
        np.random.uniform(0, 1, (64, 1, params.img_dim, params.img_dim))
    )  # want to make sure all of the inputs are positive!
    plt.title("Example random input image")
    plt.imshow(rand[0, 0, :, :])
    plt.show()
    cos_sims, lowest_cos_sims, _ = get_cosine_sims(
        rand, net, params.k_min, params.norm_addresses
    )
    plt.title("Lowest Top-k Cosine Sims for all random images.")
    plt.hist(lowest_cos_sims.flatten().numpy(), bins=50)
    plt.show()

    plt.hist(cos_sims[0, :].flatten().numpy(), bins=50)
    plt.title("All neuron Cosine Sims for a single random input")
    plt.show()

    ## Finding the top-k neurons that are the closest, using each neuron as a center point.

    neuron_addresses = net.fc1.detach().numpy()  # neurons x dendrites

    with torch.no_grad():
        norm_addresses = neuron_addresses / torch.norm(
            neuron_addresses, dim=1, keepdim=True
        )  # using tensor version as I already have this norm code for it.

        cosine_sims = norm_addresses @ norm_addresses.T

        vals, inds = torch.topk(
            cosine_sims, params.k_min + 1, dim=1
        )  # +1 to account for self match.
        lowest_cos_sims, _ = torch.min(vals, dim=1)
        closest_top_k_ind = np.argmax(lowest_cos_sims)
        print("lowest top-k cosine sim", lowest_cos_sims[closest_top_k_ind])
        plt.hist(lowest_cos_sims.flatten().numpy(), bins=100)
        plt.title("Lowest top-k cosine similarities centered on each neuron")
        plt.show()
        closest_top_k_neuron = neuron_addresses[closest_top_k_ind]
    plt.figure()
    plt.imshow(closest_top_k_neuron.reshape((params.img_dim, params.img_dim)))
    plt.title("The highest density region neuron as a centroid ")
    plt.show()

    # Converting to Hamming distances:

    # map these hamming distances/cosine similarities into exponential betas.
    # then I have an RBF kernel and standard deviation.

    lowest_hamming_dists = (
        params.input_size * (1 - lowest_cos_sims.flatten().numpy()) / 2
    )
    out = plt.hist(lowest_hamming_dists, bins=100)
    plt.xlabel("Hamming dists")
    plt.show()
    print("Smallest hamming distance", np.min(lowest_hamming_dists))
    return np.min(lowest_hamming_dists)


def pca_umap_analysis(data, color_labels, title):

    # initializing the pca
    from sklearn import decomposition
    import umap

    pca = decomposition.PCA()
    # PCA for dimensionality redcution (non-visualization)
    pca.n_components = 10
    pca_res = pca.fit_transform(data)
    percentage_var_explained = pca.explained_variance_ / np.sum(pca.explained_variance_)
    cum_var_explained = np.cumsum(percentage_var_explained)

    """plt.figure(1, figsize=(6, 4))
    plt.clf()
    plt.plot(cum_var_explained, linewidth=2)
    plt.axis('tight')
    plt.grid()
    plt.xlabel('n_components')
    plt.ylabel('Cumulative_explained_variance')
    plt.title("PCA Variance Explained - "+title)
    plt.show()"""

    plt.figure()
    plt.scatter(pca_res[:, 0], pca_res[:, 1], c=color_labels)
    plt.title("PCA Embedding - " + title)
    plt.show()

    # UMAP
    reducer = umap.UMAP()  # parallel=False
    embedding = reducer.fit_transform(data)
    plt.figure()
    plt.scatter(embedding[:, 0], embedding[:, 1], c=color_labels)
    plt.title("UMAP Embedding - " + title)
    plt.show()

    return embedding


def fancy_umap_embed(
    neuron_addresses,
    neuron_max_activations,
    embedding,
    img_dim,
    save_str=None,
    most_active=False,
):
    numbers = np.arange(10)

    plt.figure(1, figsize=(12, 8))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=neuron_max_activations)
    ax = plt.gca()
    if most_active:
        plt.title("Only Most Active - Neuron Embeddings")
        if save_str:
            save_str = save_str + "OnlyMostActive_EmbedNeuronInsets.png"
    else:
        plt.title("Neuron Embeddings")
        if save_str:
            save_str = save_str + "EmbedNeuronInsets.png"

    for n in numbers:
        if (n == neuron_max_activations).sum() == 0:
            # no matches skip this number
            continue
        matches = np.arange(neuron_addresses.shape[0])[n == neuron_max_activations]
        neuron_ind = np.random.choice(matches, 1)[0]
        dendrites = neuron_addresses[neuron_ind, :]
        embed_loc = embedding[neuron_ind, :]
        plt.scatter(embed_loc[0], embed_loc[1], c="red")
        im = OffsetImage(dendrites.reshape((img_dim, img_dim)), zoom=0.95)
        ab = AnnotationBbox(
            im, (embed_loc[0] + 0.6, embed_loc[1]), xycoords="data", frameon=True
        )
        ax.add_artist(ab)
        ax.update_datalim(np.column_stack([embed_loc[0], embed_loc[1]]))
        ax.autoscale()
    if save_str:
        plt.gcf().savefig(save_str, dpi=200)

    plt.show()

    r = np.arange(10)
    plt.scatter(r, r, c=r)
    plt.show()
