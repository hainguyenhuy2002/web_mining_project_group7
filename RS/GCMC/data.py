"""MovieLens dataset"""
import os
import re

import dgl
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch as th
from dgl.data.utils import download, extract_archive, get_download_dir
from utils import to_etype_name


class MovieLens(object):
    """MovieLens dataset used by GCMC model

    TODO(minjie): make this dataset more general

    The dataset stores MovieLens ratings in two types of graphs. The encoder graph
    contains rating value information in the form of edge types. The decoder graph
    stores plain user-movie pairs in the form of a bipartite graph with no rating
    information. All graphs have two types of nodes: "user" and "movie".

    The training, validation and test set can be summarized as follows:

    training_enc_graph : training user-movie pairs + rating info
    training_dec_graph : training user-movie pairs
    valid_enc_graph : training user-movie pairs + rating info
    valid_dec_graph : validation user-movie pairs
    test_enc_graph : training user-movie pairs + validation user-movie pairs + rating info
    test_dec_graph : test user-movie pairs

    Attributes
    ----------
    train_enc_graph : dgl.DGLGraph
        Encoder graph for training.
    train_dec_graph : dgl.DGLGraph
        Decoder graph for training.
    train_labels : torch.Tensor
        The categorical label of each user-movie pair
    train_truths : torch.Tensor
        The actual rating values of each user-movie pair
    valid_enc_graph : dgl.DGLGraph
        Encoder graph for validation.
    valid_dec_graph : dgl.DGLGraph
        Decoder graph for validation.
    valid_labels : torch.Tensor
        The categorical label of each user-movie pair
    valid_truths : torch.Tensor
        The actual rating values of each user-movie pair
    test_enc_graph : dgl.DGLGraph
        Encoder graph for test.
    test_dec_graph : dgl.DGLGraph
        Decoder graph for test.
    test_labels : torch.Tensor
        The categorical label of each user-movie pair
    test_truths : torch.Tensor
        The actual rating values of each user-movie pair
    user_feature : torch.Tensor
        User feature tensor. If None, representing an identity matrix.
    movie_feature : torch.Tensor
        Movie feature tensor. If None, representing an identity matrix.
    possible_rating_values : np.ndarray
        Available rating values in the dataset

    Parameters
    ----------
    name : str
        Dataset name. Could be "ml-100k", "ml-1m", "ml-10m"
    device : torch.device
        Device context
    mix_cpu_gpu : boo, optional
        If true, the ``user_feature`` attribute is stored in CPU
    use_one_hot_fea : bool, optional
        If true, the ``user_feature`` attribute is None, representing an one-hot identity
        matrix. (Default: False)
    symm : bool, optional
        If true, the use symmetric normalize constant. Otherwise, use left normalize
        constant. (Default: True)
    test_ratio : float, optional
        Ratio of test data
    valid_ratio : float, optional
        Ratio of validation data

    """

    def __init__(
        self,
        name,
        device,
        mix_cpu_gpu=False,
        use_one_hot_fea=False,
        symm=True,
        test_ratio=0.1,
        valid_ratio=0.1,
    ):
        self._name = name
        self._device = device
        self._symm = symm
        self._test_ratio = test_ratio
        self._valid_ratio = valid_ratio
        # download and extract
        self.user_info = pd.read_csv('data/users.csv')
        self.movie_info = pd.read_csv('data/items.csv')
        print("......")
        self.all_train_rating_info = pd.read_csv('data/train.csv')
        self.test_rating_info = pd.read_csv('data/test.csv')
        self.all_rating_info = pd.concat(
            [self.all_train_rating_info, self.test_rating_info]
        )
        print("......")
        self.train_rating_info = self.all_train_rating_info.copy()
        self.valid_rating_info = self.test_rating_info.copy()
        self.possible_rating_values = np.unique(
            self.train_rating_info["rating"].values
        )
        print("All rating pairs : {}".format(self.all_rating_info.shape[0]))
        print(
            "\tAll train rating pairs : {}".format(
                self.all_train_rating_info.shape[0]
            )
        )
        print(
            "\t\tTrain rating pairs : {}".format(
                self.train_rating_info.shape[0]
            )
        )
        print(
            "\t\tValid rating pairs : {}".format(
                self.valid_rating_info.shape[0]
            )
        )
        print(
            "\tTest rating pairs  : {}".format(self.test_rating_info.shape[0])
        )

        # Map user/movie to the global id
        self.global_user_id_map = {
            ele: i for i, ele in enumerate(self.user_info["user_id"])
        }
        self.global_movie_id_map = {
            ele: i for i, ele in enumerate(self.movie_info["item_id"])
        }
        print(
            "Total user number = {}, movie number = {}".format(
                len(self.global_user_id_map), len(self.global_movie_id_map)
            )
        )
        self._num_user = len(self.global_user_id_map)
        self._num_movie = len(self.global_movie_id_map)

        ### Generate features
        if use_one_hot_fea:
            self.user_feature = None
            self.movie_feature = None
        else:
            # if mix_cpu_gpu, we put features in CPU
            if mix_cpu_gpu:
                self.user_feature = th.FloatTensor(self._process_user_fea())
                self.movie_feature = th.FloatTensor(self._process_movie_fea())
            else:
                self.user_feature = th.FloatTensor(self._process_user_fea()).to(
                    self._device
                )
                self.movie_feature = th.FloatTensor(
                    self._process_movie_fea()
                ).to(self._device)
        if self.user_feature is None:
            self.user_feature_shape = (self.num_user, self.num_user)
            self.movie_feature_shape = (self.num_movie, self.num_movie)
        else:
            self.user_feature_shape = self.user_feature.shape
            self.movie_feature_shape = self.movie_feature.shape
        info_line = "Feature dim: "
        info_line += "\nuser: {}".format(self.user_feature_shape)
        info_line += "\nmovie: {}".format(self.movie_feature_shape)
        print(info_line)

        (
            all_train_rating_pairs,
            all_train_rating_values,
        ) = self._generate_pair_value(self.all_train_rating_info)
        train_rating_pairs, train_rating_values = self._generate_pair_value(
            self.train_rating_info
        )
        valid_rating_pairs, valid_rating_values = self._generate_pair_value(
            self.valid_rating_info
        )
        test_rating_pairs, test_rating_values = self._generate_pair_value(
            self.test_rating_info
        )

        def _make_labels(ratings):
            labels = th.LongTensor(
                np.searchsorted(self.possible_rating_values, ratings)
            ).to(device)
            return labels

        self.train_enc_graph = self._generate_enc_graph(
            train_rating_pairs, train_rating_values, add_support=True
        )
        self.train_dec_graph = self._generate_dec_graph(train_rating_pairs)
        self.train_labels = _make_labels(train_rating_values)
        self.train_truths = th.FloatTensor(train_rating_values).to(device)

        self.valid_enc_graph = self.train_enc_graph
        self.valid_dec_graph = self._generate_dec_graph(valid_rating_pairs)
        self.valid_labels = _make_labels(valid_rating_values)
        self.valid_truths = th.FloatTensor(valid_rating_values).to(device)

        self.test_enc_graph = self._generate_enc_graph(
            all_train_rating_pairs, all_train_rating_values, add_support=True
        )
        self.test_dec_graph = self._generate_dec_graph(test_rating_pairs)
        self.test_labels = _make_labels(test_rating_values)
        self.test_truths = th.FloatTensor(test_rating_values).to(device)

        def _npairs(graph):
            rst = 0
            for r in self.possible_rating_values:
                r = to_etype_name(r)
                rst += graph.num_edges(str(r))
            return rst

        print(
            "Train enc graph: \t#user:{}\t#movie:{}\t#pairs:{}".format(
                self.train_enc_graph.num_nodes("user"),
                self.train_enc_graph.num_nodes("movie"),
                _npairs(self.train_enc_graph),
            )
        )
        print(
            "Train dec graph: \t#user:{}\t#movie:{}\t#pairs:{}".format(
                self.train_dec_graph.num_nodes("user"),
                self.train_dec_graph.num_nodes("movie"),
                self.train_dec_graph.num_edges(),
            )
        )
        print(
            "Valid enc graph: \t#user:{}\t#movie:{}\t#pairs:{}".format(
                self.valid_enc_graph.num_nodes("user"),
                self.valid_enc_graph.num_nodes("movie"),
                _npairs(self.valid_enc_graph),
            )
        )
        print(
            "Valid dec graph: \t#user:{}\t#movie:{}\t#pairs:{}".format(
                self.valid_dec_graph.num_nodes("user"),
                self.valid_dec_graph.num_nodes("movie"),
                self.valid_dec_graph.num_edges(),
            )
        )
        print(
            "Test enc graph: \t#user:{}\t#movie:{}\t#pairs:{}".format(
                self.test_enc_graph.num_nodes("user"),
                self.test_enc_graph.num_nodes("movie"),
                _npairs(self.test_enc_graph),
            )
        )
        print(
            "Test dec graph: \t#user:{}\t#movie:{}\t#pairs:{}".format(
                self.test_dec_graph.num_nodes("user"),
                self.test_dec_graph.num_nodes("movie"),
                self.test_dec_graph.num_edges(),
            )
        )

    def _generate_pair_value(self, rating_info):
        rating_pairs = (
            np.array(
                [
                    self.global_user_id_map[ele]
                    for ele in rating_info["user_id"]
                ],
                dtype=np.int64,
            ),
            np.array(
                [
                    self.global_movie_id_map[ele]
                    for ele in rating_info["item_id"]
                ],
                dtype=np.int64,
            ),
        )
        rating_values = rating_info["rating"].values.astype(np.float32)
        return rating_pairs, rating_values

    def _generate_enc_graph(
        self, rating_pairs, rating_values, add_support=False
    ):
        user_movie_R = np.zeros(
            (self._num_user, self._num_movie), dtype=np.float32
        )
        user_movie_R[rating_pairs] = rating_values

        data_dict = dict()
        num_nodes_dict = {"user": self._num_user, "movie": self._num_movie}
        rating_row, rating_col = rating_pairs
        for rating in self.possible_rating_values:
            ridx = np.where(rating_values == rating)
            rrow = rating_row[ridx]
            rcol = rating_col[ridx]
            rating = to_etype_name(rating)
            data_dict.update(
                {
                    ("user", str(rating), "movie"): (rrow, rcol),
                    ("movie", "rev-%s" % str(rating), "user"): (rcol, rrow),
                }
            )
        graph = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)

        # sanity check
        assert (
            len(rating_pairs[0])
            == sum([graph.num_edges(et) for et in graph.etypes]) // 2
        )

        if add_support:

            def _calc_norm(x):
                x = x.numpy().astype("float32")
                x[x == 0.0] = np.inf
                x = th.FloatTensor(1.0 / np.sqrt(x))
                return x.unsqueeze(1)

            user_ci = []
            user_cj = []
            movie_ci = []
            movie_cj = []
            for r in self.possible_rating_values:
                r = to_etype_name(r)
                user_ci.append(graph["rev-%s" % r].in_degrees())
                movie_ci.append(graph[r].in_degrees())
                if self._symm:
                    user_cj.append(graph[r].out_degrees())
                    movie_cj.append(graph["rev-%s" % r].out_degrees())
                else:
                    user_cj.append(th.zeros((self.num_user,)))
                    movie_cj.append(th.zeros((self.num_movie,)))
            user_ci = _calc_norm(sum(user_ci))
            movie_ci = _calc_norm(sum(movie_ci))
            if self._symm:
                user_cj = _calc_norm(sum(user_cj))
                movie_cj = _calc_norm(sum(movie_cj))
            else:
                user_cj = th.ones(
                    self.num_user,
                )
                movie_cj = th.ones(
                    self.num_movie,
                )
            graph.nodes["user"].data.update({"ci": user_ci, "cj": user_cj})
            graph.nodes["movie"].data.update({"ci": movie_ci, "cj": movie_cj})

        return graph

    def _generate_dec_graph(self, rating_pairs):
        ones = np.ones_like(rating_pairs[0])
        user_movie_ratings_coo = sp.coo_matrix(
            (ones, rating_pairs),
            shape=(self.num_user, self.num_movie),
            dtype=np.float32,
        )
        g = dgl.bipartite_from_scipy(
            user_movie_ratings_coo, utype="_U", etype="_E", vtype="_V"
        )
        return dgl.heterograph(
            {("user", "rate", "movie"): g.edges()},
            num_nodes_dict={"user": self.num_user, "movie": self.num_movie},
        )

    @property
    def num_links(self):
        return self.possible_rating_values.size

    @property
    def num_user(self):
        return self._num_user

    @property
    def num_movie(self):
        return self._num_movie

    def _load_raw_rates(self, file_path, sep):
        """In MovieLens, the rates have the following format

        ml-100k
        user id \t movie id \t rating \t timestamp

        ml-1m/10m
        UserID::MovieID::Rating::Timestamp

        timestamp is unix timestamp and can be converted by pd.to_datetime(X, unit='s')

        Parameters
        ----------
        file_path : str

        Returns
        -------
        rating_info : pd.DataFrame
        """
        rating_info = pd.read_csv(
            file_path,
            sep=sep,
            header=None,
            names=["user_id", "movie_id", "rating", "timestamp"],
            dtype={
                "user_id": np.int32,
                "movie_id": np.int32,
                "ratings": np.float32,
                "timestamp": np.int64,
            },
            engine="python",
        )
        return rating_info

    def _process_user_fea(self):
        """

        Parameters
        ----------
        user_info : pd.DataFrame
        name : str
        For ml-100k and ml-1m, the column name is ['id', 'gender', 'age', 'occupation', 'zip_code'].
            We take the age, gender, and the one-hot encoding of the occupation as the user features.
        For ml-10m, there is no user feature and we set the feature to be a single zero.

        Returns
        -------
        user_features : np.ndarray

        """
        if self._name == "ml-100k" or self._name == "ml-1m":
            ages = self.user_info["age"].values.astype(np.float32)
            gender = (self.user_info["gender"] == "F").values.astype(np.float32)
            all_occupations = set(self.user_info["occupation"])
            occupation_map = {ele: i for i, ele in enumerate(all_occupations)}
            occupation_one_hot = np.zeros(
                shape=(self.user_info.shape[0], len(all_occupations)),
                dtype=np.float32,
            )
            occupation_one_hot[
                np.arange(self.user_info.shape[0]),
                np.array(
                    [
                        occupation_map[ele]
                        for ele in self.user_info["occupation"]
                    ]
                ),
            ] = 1
            user_features = np.concatenate(
                [
                    ages.reshape((self.user_info.shape[0], 1)) / 50.0,
                    gender.reshape((self.user_info.shape[0], 1)),
                    occupation_one_hot,
                ],
                axis=1,
            )
        elif self._name == "ml-10m":
            user_features = np.zeros(
                shape=(self.user_info.shape[0], 1), dtype=np.float32
            )
        else:
            raise NotImplementedError
        return user_features

    def _process_movie_fea(self):
        """

        Parameters
        ----------
        movie_info : pd.DataFrame
        name :  str

        Returns
        -------
        movie_features : np.ndarray
            Generate movie features by concatenating embedding and the year

        """
        import torchtext
        from torchtext.data.utils import get_tokenizer

        if self._name == "ml-100k":
            GENRES = GENRES_ML_100K
        elif self._name == "ml-1m":
            GENRES = GENRES_ML_1M
        elif self._name == "ml-10m":
            GENRES = GENRES_ML_10M
        else:
            raise NotImplementedError

        # Old torchtext-legacy API commented below
        # TEXT = torchtext.legacy.data.Field(tokenize='spacy', tokenizer_language='en_core_web_sm')
        tokenizer = get_tokenizer(
            "spacy", language="en_core_web_sm"
        )  # new API (torchtext 0.9+)
        embedding = torchtext.vocab.GloVe(name="840B", dim=300)

        title_embedding = np.zeros(
            shape=(self.movie_info.shape[0], 300), dtype=np.float32
        )
        release_years = np.zeros(
            shape=(self.movie_info.shape[0], 1), dtype=np.float32
        )
        p = re.compile(r"(.+)\s*\((\d+)\)")
        for i, title in enumerate(self.movie_info["title"]):
            match_res = p.match(title)
            if match_res is None:
                print(
                    "{} cannot be matched, index={}, name={}".format(
                        title, i, self._name
                    )
                )
                title_context, year = title, 1950
            else:
                title_context, year = match_res.groups()
            # We use average of glove
            # Upgraded torchtext API:  TEXT.tokenize(title_context) --> tokenizer(title_context)
            title_embedding[i, :] = (
                embedding.get_vecs_by_tokens(tokenizer(title_context))
                .numpy()
                .mean(axis=0)
            )
            release_years[i] = float(year)
        movie_features = np.concatenate(
            (
                title_embedding,
                (release_years - 1950.0) / 100.0,
                self.movie_info[GENRES],
            ),
            axis=1,
        )
        return movie_features


if __name__ == "__main__":
    MovieLens("ml-100k", device=th.device("cpu"), symm=True)