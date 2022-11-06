import numpy as np
import more_itertools as mit

def find_epsilon(e_s, sd_lim=12.0, error_buffer = 100):

    max_score = -10000000
    mean_e_s = np.mean(e_s)
    sd_e_s = np.std(e_s)
    for z in np.arange(2.5, sd_lim, 0.5):
        epsilon = mean_e_s + (sd_e_s * z)

        pruned_e_s = e_s[e_s < epsilon]

        i_anom = np.argwhere(e_s >= epsilon).reshape(-1, )
        buffer = np.arange(1, error_buffer)
        i_anom = np.sort(np.concatenate((i_anom,
                                         np.array([i + buffer for i in i_anom])
                                         .flatten(),
                                         np.array([i - buffer for i in i_anom])
                                         .flatten())))
        i_anom = i_anom[(i_anom < len(e_s)) & (i_anom >= 0)]
        i_anom = np.sort(np.unique(i_anom))

        if len(i_anom) > 0:
            # group anomalous indices into continuous sequences
            groups = [list(group) for group
                      in mit.consecutive_groups(i_anom)]
            E_seq = [(g[0], g[-1]) for g in groups if not g[0] == g[-1]]

            mean_perc_decrease = (mean_e_s - np.mean(pruned_e_s)) \
                                 / mean_e_s
            sd_perc_decrease = (sd_e_s - np.std(pruned_e_s)) \
                               / sd_e_s
            score = (mean_perc_decrease + sd_perc_decrease) \
                    / (len(E_seq) ** 2 + len(i_anom))

            # sanity checks / guardrails
            if score >= max_score and len(E_seq) <= 5 and \
                    len(i_anom) < (len(e_s) * 0.5):
                max_score = score
                sd_threshold = z
                epsilon = mean_e_s + z * sd_e_s

    return epsilon