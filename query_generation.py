import itertools, time, typing

import numpy as np
import scipy.optimize as opt
import theano
import theano.tensor as tt
from utils import *

# import domain
# import traj
# import utils as utils


ObjectiveFunctionType = typing.Callable[
    [int, int, tt.TensorVariable, tt.TensorVariable],
    typing.List[tt.TensorVariable],
]


# def min_volume_objective(num_queries: int,
#                          num_w_samples: int,
#                          w_samples: tt.TensorVariable,
#                          traj_features: tt.TensorVariable,
#                          beta_pref: float,
#                          ) -> tt.TensorVariable:
#     # volumes_removed_list: typing.List[tt.TensorVariable] = []
#     volumes_removed_list = []
#     for i in range(num_queries):
#         feature_differences = traj_features - \
#             (traj_features[i, :] * tt.ones((num_queries, 1)))

#         # phis is num_queries by num_w_samples
#         phis = tt.dot(feature_differences, w_samples.T)
#         # exps is num_queries by num_w_samples
#         exps = tt.exp(beta_pref * phis)
#         # estimates is 1 by num_w_samples
#         estimates = 1. - 1. / tt.sum(exps, axis=0)
#         volumes_removed_list.append(tt.sum(estimates) / float(num_w_samples))
#     volumes_removed = tt.stack(volumes_removed_list)
#     return tt.min(volumes_removed)


# pick_best = min_volume_objective

# def rank_objective(num_queries: int,
#                          num_w_samples: int,
#                          samples: tt.TensorVariable,
#                          features: tt.TensorVariable,
#                          beta_pref: float) -> float:
#     """
#     The ranking maximum volume removal objective function, using the Plackett-Luce model of human behavior.

#     CANNOT BE USED WITH (INC_PREV_QUERY AND NO DEMPREF).

#     :param features: a list containing the feature values of each query.
#     :param samples: samples of w, used to approximate the objective.
#     :return: the value of the objective function, evaluated on the given queries' features.
#     """
#     # features: n_queries x feature_size
#     # samples: n_samples x feature_size
#     exp_rewards = tt.sum(tt.dot(features, samples.T), axis=1) / samples.shape[0]  # n_queries x 1 -- summed across samples
#     volumes_removed = []
#     rankings = itertools.permutations(list(range(num_queries)))  # iterating over all possible rankings
#     for rank in rankings:
#         exp_rewards_sorted = [None] * len(rank)
#         for i in range(len(rank)):
#             exp_rewards_sorted[rank[i]] = exp_rewards[i]

#         value, i = 1, 0
#         for i in range(len(rank) - 1):
#             exp_i = [beta_pref * (exp_rewards_sorted[j] - exp_rewards_sorted[i]) for j in range(i, len(rank) - 1)]
#             value *= (1. / tt.sum(tt.exp(exp_i)))
#         volumes_removed.append(1 - value)
#     return tt.min(volumes_removed)

# rank = rank_objective


def approx_min_volume_2_objective(num_queries: int,
                                  num_w_samples: int,
                                  w_samples: tt.TensorVariable,
                                  traj_features: tt.TensorVariable,
                                  beta_pref: float,
                                  ) -> tt.TensorVariable:
    assert num_queries == 2, "approx objective can only handle 2 queries"

    phis = tt.dot(traj_features[0] - traj_features[1], w_samples.T)

    posExps = tt.exp(beta_pref * phis)
    negExps = tt.exp(beta_pref * -phis)

    posEstimates = 1. - tt.min([tt.ones(num_w_samples), posExps], axis=0)
    negEstimates = 1. - tt.min([tt.ones(num_w_samples), negExps], axis=0)

    posAvg = tt.sum(posEstimates) / float(num_w_samples)
    negAvg = tt.sum(negEstimates) / float(num_w_samples)

    objs = tt.stack([posAvg, negAvg])

    return tt.min(objs)


approx = approx_min_volume_2_objective

class ApproxQueryGenerator(object):
    def __init__(self,
                #  dom: domain.Domain,
                 env,
                 num_queries: int,
                 query_length: int,
                 num_expectation_samples: int,
                 include_previous_query: bool,
                 generate_scenario: bool,
                 update_func: str,
                 beta_pref: float,
                 ) -> None:
        """
        Initializes the approx query generation, which generates queries using approx gradients.

        :param dom: the domain to generate queries on.
        :param num_queries: number of queries to generate at each time step.
        :param query_length: the length of each query.
        :param num_expectation_samples: number of samples to use in approximating the objective function.
        :param include_previous_query: boolean for whether one of the queries is the previously selected query.
        :param generate_scenario: boolean for whether we want to generate the scenario -- i.e., other agents' behavior.
        :param update_func: the update_func used; the options are "pick_best", "approx", and "rank".
        :param beta_pref: the rationality parameter for the human selecting her query.
        """
        assert num_queries >= 1, \
            "QueryGenerator.__init__: num_queries must be at least 1"
        assert query_length >= 1, \
            "QueryGenerator.__init__: query_length must be at least 1"
        assert num_expectation_samples >= 1, \
            "QueryGenerator.__init__: num_expectation_samples must be \
                at least 1"
        # self.domain = dom
        self.env = env
        self.num_queries = num_queries
        self.query_length = query_length
        self.num_expectation_samples = num_expectation_samples
        self.include_previous_query = include_previous_query
        self.generate_scenario = generate_scenario # Currently must be False
        assert self.generate_scenario == False, "Cannot generate scenario when using approximate gradients"
        self.update_func = update_func
        self.beta_pref = beta_pref
        self.num_new_queries = self.num_queries - 1 if self.include_previous_query else self.num_queries


    def queries(self, w_samples:np.ndarray, blank_traj:bool=False) -> typing.List[typing.List]:
        """
        Generates self.num_queries number of queries, that (locally) maximize the maximum volume removal objective.

        :param w_samples: Samples of w.
        :param last_query: The previously selected query. Only required if self.inc_prev_query is True.
        :param blank_traj: True is last_query is blank. (Only True if not using Dempref but using inc_prev_traj.)
        :return: a list of trajectories (queries).
        """
        start = time.time()

        def func(controls: np.ndarray, *args) -> float:
            """
            The function to be minimized by L_BFGS.

            :param controls: an array, concatenated to contain the control input for all queries.
            :param args: the first argument is the domain, and the second is the samples that will be used to
                approximate the objective function.
            :return: the value of the objective function for the given set of controls.
            """
            # domain = args[0]
            samples = args[0]
            features = generate_features(controls)
            if self.update_func == "pick_best":
                return -objective(features, samples)
            elif self.update_func == "approx":
                return -approx_objective(features, samples)
            # else:
            #     return -rank_objective(features, samples)

        def generate_features(controls: np.ndarray) -> typing.List:
            """
            Generates a set of features for the set of controls provided.

            :param domain: the domain that the queries are being generated on.
            :param controls: an array, concatenated to contain the control input for all queries.
            :param last_query: the last query chosen by the human. Only required if self.inc_prev_query is true.
            :return: a list containing the feature values of each query.
            """
            z = self.query_length #* self.env.action_space.shape[0]
            controls = np.array(controls)
            controls_set = [controls[i * z: (i + 1) * z] for i in range(self.num_new_queries)]
            trajs = [dempref_run(self.env, c)[0] for c in controls_set]
            # print(trajs[0])
            features = [np_features(traj, dempref=True) for traj in trajs]
            # print("gen features: ", features)
            return features

        def objective(features: typing.List, samples:np.ndarray) -> float:
            """
            The standard maximum volume removal objective function.

            :param features: a list containing the feature values of each query.
            :param samples: samples of w, used to approximate the objective.
            :return: the value of the objective function, evaluated on the given queries' features.
            """
            volumes_removed = []
            for i in range(len(features)):
                feature_diff = np.array([f - features[i] for f in features]) # n_queries x feature_size
                
                # import pdb; pdb.set_trace()
                # weighted_feature_diff = np.sum(np.dot(feature_diff, samples.T), axis=1)/samples.shape[0] # n_queries x 1 -- summed across samples
                weighted_feature_diff = np.dot(feature_diff, samples.T)/samples.shape[0]
                v_removed = 1. - 1./ np.sum(np.exp(self.beta_pref * weighted_feature_diff))
                volumes_removed.append(v_removed)
            return np.min(volumes_removed)

        def approx_objective(features, samples) -> float:
            """
            The approximate maximum volume removal objective function.

            :param features: a list containing the feature values of each query.
            :param samples: samples of w, used to approximate the objective.
            :return: the value of the objective function, evaluated on the given queries' features.
            """
            volumes_removed = []
            for i in range(len(features)):
                feature_diff = features[i] - features[1-i] # 1 x feature_size
                weighted_feature_diff = np.sum(np.dot(feature_diff, samples.T))/samples.shape[0] # 1 x 1 -- summed across samples
                v_removed = 1. - np.minimum(1., np.exp(self.beta_pref * weighted_feature_diff))
                volumes_removed.append(v_removed)
            print(volumes_removed)
            return np.min(volumes_removed)

        # def rank_objective(features, samples) -> float:
        #     """
        #     The ranking maximum volume removal objective function, using the Plackett-Luce model of human behavior.

        #     CANNOT BE USED WITH (INC_PREV_QUERY AND NO DEMPREF).

        #     :param features: a list containing the feature values of each query.
        #     :param samples: samples of w, used to approximate the objective.
        #     :return: the value of the objective function, evaluated on the given queries' features.
        #     """
        #     # features: n_queries x feature_size
        #     # samples: n_samples x feature_size
        #     exp_rewards = np.sum(np.dot(features, samples.T), axis=1)/samples.shape[0] # n_queries x 1 -- summed across samples
        #     volumes_removed = []
        #     rankings = itertools.permutations(list(range(self.num_queries))) # iterating over all possible rankings
        #     for rank in rankings:
        #         exp_rewards_sorted = [None] * len(rank)
        #         for i in range(len(rank)):
        #             exp_rewards_sorted[rank[i]] = exp_rewards[i]

        #         value, i = 1, 0
        #         for i in range(len(rank) - 1):
        #             value *= (1. / np.sum(np.exp(self.beta_pref * (np.array(exp_rewards_sorted[i:]) - exp_rewards_sorted[i]))))
        #         volumes_removed.append(1 - value)
        #     return np.min(volumes_removed)


        z = self.query_length #* self.env.action_space.shape[0]
        # control_bounds = [(self.env.action_space.low[i], self.env.action_space.high[i]) for i in range(self.env.action_space.shape[0])]
        lower_input_bound = [0] * self.query_length
        upper_input_bound = [4] * self.query_length
        opt_res = opt.fmin_l_bfgs_b(func,
                                    x0=np.random.uniform(low=lower_input_bound*self.num_new_queries , high= upper_input_bound*self.num_new_queries,#self.num_new_queries*lower_input_bound, high=self.num_new_queries*upper_input_bound,
                                                         size=(self.num_new_queries * z)), args=(w_samples),
                                                         bounds=[(0,4)]*self.num_new_queries*self.query_length,
                                                         approx_grad=True)
        query_controls = [opt_res[0][i * z: (i + 1) * z] for i in range(self.num_new_queries)]
        end = time.time()
        # print(f"Finished computing {len(query_controls)} queries in " + str(end - start) + "s")
        # if self.include_previous_query and not blank_traj:
        #     return [last_query] + [self.domain.run(c) for c in query_controls]
        # else:
        # print(len(query_controls[0]))
        return [dempref_run(self.env, c)[0] for c in query_controls], query_controls


ScenarioRequired = "QueryGenerator.queries: keyword argument \
                    'scenario' is required."
ScenarioDisabled = "QueryGenerator.queries: keyword argument \
                    'scenario' is disabled."
ScenarioLengthMismatch = "QueryGenerator.queries: 'scenarios' \
                            length must be query_length"
ScenarioNumAgentsMismatch = "QueryGenerator.queries: 'scenarios' \
                                num_agents must be num_agents"


# def sum_trajectory_features(d: domain.Domain,
#                             query_length: int,
#                             human_xs,
#                             other_xs
#                             ):
#     assert len(other_xs) > 0, \
#         "sum_trajectory_features: other_xs should have positive length"
#     assert len(other_xs[0]) == d.num_others, \
#         "sum_trajectory_features: other_xs num_others should match the domain"

#     x = tt.stack([d.features_function(human_xs[t],
#                                       [other_xs[t][i]
#                                        for i in range(d.num_others)]
#                                       )
#                   for t in range(query_length)])
#     return tt.sum(x, axis=0)
