from collections import Counter
from .CommonFunctions import time_wrapper, create_custom_logger


class InformationCalculator:
    def __init__(self):
        self.local_logger = create_custom_logger(InformationCalculator.__name__)
        self.interpolation_smooth_threshold = 0.0001
        self.turing_smooth_threshold = 0.05
        self.top_k = 10

    @time_wrapper
    def generate_count_per_gram_type(self, inp_list: list, gram_type: int):
        all_phrase_list = []
        for ind_sen in inp_list:
            phrase_list = ind_sen.split(" ")
            if len(phrase_list) >= gram_type:
                all_phrase_list.append(" ".join(["<S>"] + phrase_list[0: gram_type - 1]))
                for idx in range(0, len(phrase_list) - gram_type + 1):
                    all_phrase_list.append(" ".join(phrase_list[idx: idx + gram_type]))
        ngram_to_count_dict = Counter(all_phrase_list)
        return ngram_to_count_dict

    def count_ngrams(self, inp_list: list, ngramtype: list):
        ngram_list = []
        for ngram in ngramtype:
            ngram_list.append(self.generate_count_per_gram_type(inp_list, ngram))
        return ngram_list

    @time_wrapper
    def generate_backoff_and_interpolation(self, unigram_to_count_dict: dict,
                                           bi_gram_to_count_dict: dict,
                                           inp_list: list):
        prob_sen_list = []
        alpha_1, alpha_2 = 0.1, 0.9
        total_words_count = 0
        for ind_phrase in unigram_to_count_dict:
            total_words_count += unigram_to_count_dict[ind_phrase]

        for ind_inp in inp_list:
            phrase_list = ind_inp.split(" ")
            phrase_list.insert(0, "<S>")
            final_prob = 1
            for idx in range(0, len(phrase_list) - 1):
                unigram_prob = unigram_to_count_dict[phrase_list[idx]] / total_words_count
                bigram_prob = bi_gram_to_count_dict[" ".join(phrase_list[idx: idx + 2])] / unigram_to_count_dict[
                    phrase_list[idx]] if len(phrase_list) >= 2 else 0
                final_prob *= alpha_1 * unigram_prob + alpha_2 * bigram_prob
            prob_sen_list.append(final_prob)
        return prob_sen_list

    @time_wrapper
    def generate_good_turing(self, unigram_to_count_dict: dict, bi_gram_to_count_dict: dict, inp_list: list):
        prob_sen_list = []
        bi_gram_fre_count, unigram_fre_count = {}, {}
        total_bi_grams_count, total_unigram_cnt, max_cnt = 0, 0, -1
        for cnt in bi_gram_to_count_dict.values():
            if cnt not in bi_gram_fre_count:
                bi_gram_fre_count[cnt] = 0
            bi_gram_fre_count[cnt] += 1
            total_bi_grams_count += cnt
            if max_cnt < cnt:
                max_cnt = cnt
        for idx in range(0, max_cnt + 1):
            if idx not in bi_gram_fre_count:
                bi_gram_fre_count[idx] = 1
        max_cnt = -1
        for cnt in unigram_to_count_dict.values():
            if cnt not in unigram_fre_count:
                unigram_fre_count[cnt] = 0
            unigram_fre_count[cnt] += 1
            total_unigram_cnt += cnt
            if max_cnt < cnt:
                max_cnt = cnt
        for idx in range(0, max_cnt + 1):
            if idx not in unigram_fre_count:
                unigram_fre_count[idx] = 1
        #print(bi_gram_fre_count)
        #print(unigram_fre_count)
        for inp in inp_list:
            phrase_list = inp.split(" ")
            final_prob = 1
            if len(phrase_list) >= 2:
                for idx in range(0, len(phrase_list) - 1):
                    bi_gram_phrase = " ".join(phrase_list[idx: idx + 2])
                    localcnt = bi_gram_to_count_dict.get(bi_gram_phrase, -1)
                    if localcnt != -1:
                        final_prob *= ((localcnt + 1) *
                                       (bi_gram_fre_count[localcnt] / bi_gram_fre_count[localcnt + 1])) \
                                      / total_bi_grams_count
                    else:
                        final_prob *= bi_gram_fre_count[0] / total_bi_grams_count
            else:
                for idx in range(0, len(phrase_list)):
                    unigram_phrase = " ".join(phrase_list[idx: idx + 1])
                    localcnt = unigram_to_count_dict.get(unigram_phrase, -1)
                    if localcnt != -1:
                        final_prob *= ((localcnt + 1) *
                                       (unigram_fre_count[localcnt] / unigram_fre_count[localcnt + 1])) \
                                      / total_unigram_cnt
                    else:
                        final_prob *= unigram_fre_count[0] / total_unigram_cnt
            prob_sen_list.append(final_prob)
        return prob_sen_list

    @time_wrapper
    def generate_and_sort_probability_for_each_type(self, inp_list):
        flattened_list = [inp_info for outer_list in inp_list for inp_info in outer_list]
        self.local_logger.info("Probability generation starting for all phrases from conversations")
        self.local_logger.debug("Generating ngrams for the flattened list")
        ngram_list = self.count_ngrams(flattened_list, [1, 2])
        unique_items_list = list(set(flattened_list))
        #self.local_logger.debug(f"Unique Items:{unique_items_list}")
        self.local_logger.debug("Generating probabilities for phrases through smoothing techniques")
        interpol_theme_prob_list = self.generate_backoff_and_interpolation(ngram_list[0], ngram_list[1],
                                                                           unique_items_list)
        turing_theme_prob_list = self.generate_good_turing(ngram_list[0], ngram_list[1], unique_items_list)
        interpol_theme_score_list = list(zip(unique_items_list, interpol_theme_prob_list))
        turing_theme_score_list = list(zip(unique_items_list, turing_theme_prob_list))
        sorted_interpol_theme_list = sorted(interpol_theme_score_list, key=lambda info: info[1], reverse=True)
        sorted_turing_theme_list = sorted(turing_theme_score_list, key=lambda info: info[1], reverse=True)
        self.local_logger.debug("Choosing important phrases based on probabilities")
        imp_phrase_list = []
        [imp_phrase_list.append(phrase_info[0]) for phrase_info in sorted_interpol_theme_list[:10]
         if phrase_info[1] >= self.interpolation_smooth_threshold and phrase_info[0] not in imp_phrase_list]
        [imp_phrase_list.append(phrase_info[0]) for phrase_info in sorted_turing_theme_list
         if phrase_info[1] >= self.turing_smooth_threshold and phrase_info[0] not in imp_phrase_list]
        return imp_phrase_list

