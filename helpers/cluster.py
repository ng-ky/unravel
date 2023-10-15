import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_samples, silhouette_score


class Cluster():
    def __init__(self, input_csv, output, pca_value):
        df = pd.read_csv(input_csv)
        self.df = df.fillna(0)
        # remove the first column "path"
        X = self.df.iloc[:,1:]
        if pca_value > 0:
            pca = PCA(n_components=pca_value, random_state=42)
            pca.fit(X)
            self.X = pca.transform(X)
        else:
            self.X = X
        self.output = output


    def predict(self, k):
        print (f"[+] Split into {k} clusters.")
        kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
        all_predictions = kmeans.fit_predict(self.X)
        self.df['prediction'] = all_predictions
        self.df.to_csv(self.output, index=False)


    def analyse(self, max_k):
        print (f"[+] Generate data for {max_k} clusters.")

        k_range = range(2, max_k)
        inertias = []
        silhouette_scores = []
        for i in k_range:
            kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42, n_init=10)
            #km = kmeans.fit(self.X)
            labels = kmeans.fit_predict(self.X)
            inertias.append(kmeans.inertia_)
            #silhouette_scores.append(silhouette_score(self.X, km.labels_))
            silhouette_scores.append(silhouette_score(self.X, labels, metric='euclidean'))
        self.plot(max_k, inertias, silhouette_scores)


    def plot(self, max_k, inertias, silhouette_score):
        x = np.arange(1, max_k, 1)

        figure, axis = plt.subplots(1, 2)

        figure.set_figwidth(12)

        axis[0].plot(range(2, max_k), inertias)
        axis[0].set_title('The Elbow method')
        axis[0].set_xlabel("Number of clusters")
        axis[0].set_ylabel("Within-Cluster Sum of Squares (WCSS)")
        axis[0].set_xticks(x)
        axis[0].grid()

        axis[1].plot(range(2, max_k), silhouette_score)
        axis[1].set_title('The Silhouette Coefficient')
        axis[1].set_xlabel("Number of clusters")
        axis[1].set_ylabel("Score")
        axis[1].set_xticks(x)
        axis[1].grid()

        plt.savefig(self.output)


    def analyse_silhouette(self, max_k):
        k_range = range(2, max_k)
        X = self.X

        scores = []
        for n_clusters in k_range:
            # Create a subplot with 1 row and 2 columns
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.set_size_inches(18, 5)

            # The 1st subplot is the silhouette plot.
            # The silhouette coefficient can range from -1 to 1.
            ax1.set_xlim([-0.5, 1])
            # The (n_clusters+1)*10 is for inserting blank space between silhouette
            # plots of individual clusters, to demarcate them clearly.
            ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

            # Initialize the clusterer with n_clusters value and a random_state for reproducibility.
            clusterer = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
            cluster_labels = clusterer.fit_predict(X)

            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed clusters.
            silhouette_avg = silhouette_score(X, cluster_labels)
            scores.append(silhouette_avg)

            title = "The silhouette score for {n} clusters is {score}.".format(n=n_clusters, score=float('%.4g' % silhouette_avg))
            # TODO round off sil score to 3 d.p.

            # Compute the silhouette scores for each sample
            sample_silhouette_values = silhouette_samples(X, cluster_labels)

            y_lower = 10
            for i in range(n_clusters):
                # Aggregate the silhouette scores for samples belonging to cluster i, and sort them.
                ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
                ith_cluster_silhouette_values.sort()
                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i
                color = cm.nipy_spectral(float(i) / n_clusters)
                ax1.fill_betweenx(
                    np.arange(y_lower, y_upper),
                    0,
                    ith_cluster_silhouette_values,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.7,
                )
                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_title("The silhouette plot for {n} clusters.".format(n=n_clusters))
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

            # 2nd Plot to visualize the clusters formed
            perplexity = np.minimum(X.shape[0] - 1, 30)
            tsne = TSNE(n_components=2, random_state=42, init='random', perplexity=perplexity)
            z = tsne.fit_transform(X)

            colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
            ax2.scatter(z[:, 0], z[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k")

            ax2.set_title("tSNE projection for {i} clusters.".format(i=n_clusters))
            ax2.set_xlabel("component-1")
            ax2.set_ylabel("component-2")
            #ax2.legend()

            plt.suptitle(
                title,
                fontsize=14,
                fontweight="bold",
            )

            filename = "{}-{}.jpg".format(self.output, n_clusters)
            plt.savefig(filename)

        # clear the current figure
        plt.clf()

        # plot the graph of silhouette scores
        plt.figure(figsize=(5,5))
        plt.plot(k_range, scores)
        plt.title('Silhouette Score vs Number of Clusters')
        plt.xlabel("Number of clusters")
        plt.ylabel("Score")
        plt.xticks(np.arange(2, max_k, 1))
#        plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
        plt.grid()
        plt.savefig('{}-scores.jpg'.format(self.output))


    def create(self, k):
        print (f"[+] Create subfolders for {k} clusters.")
        if not os.path.exists(self.output):
            os.mkdir(self.output)
            for index in range(k):
                cluster_folder = os.path.join(self.output, "{:02d}".format(index))
                os.mkdir(cluster_folder)
                for bitmap_path in self.df[self.df['prediction'] == index]['path'].to_list():
                    src = bitmap_path
                    dst = os.path.join(cluster_folder, os.path.basename(bitmap_path))
                    #print (src, dst)
                    os.symlink(src, dst)
        else:
            print (f"[-] {self.output} already exists. Choose an empty one.")


