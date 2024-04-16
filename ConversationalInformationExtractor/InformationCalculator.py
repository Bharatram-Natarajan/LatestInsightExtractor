from collections import Counter
#from .CommonFunctions import time_wrapper, create_custom_logger


class InformationCalculator:
    def __init__(self):
        self.local_logger = create_custom_logger(InformationCalculator.__name__)
        self.add_one_smooth_threshold = 0.0001
        self.interpolation_smooth_threshold = 0.0001
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
    def generate_add_one_smoothing(self, unigram_to_count_dict: dict,
                                   bi_gram_to_count_dict: dict,
                                   inp_list: list):
        prob_sen_list = []
        total_words_count = 0
        for ind_phrase in unigram_to_count_dict:
            total_words_count += unigram_to_count_dict[ind_phrase]
        for ind_inp in inp_list:
            phrase_list = ind_inp.split(" ")
            if len(phrase_list) >= 2:
                final_prob = 1 * (bi_gram_to_count_dict[" ".join(["<S>"] + phrase_list[0:1])] + 1 / (
                            unigram_to_count_dict["<S>"] + total_words_count))
                for idx in range(0, len(phrase_list) - 1):
                    ind_phrase = " ".join(phrase_list[idx: idx + 2])
                    final_prob = final_prob * (bi_gram_to_count_dict[ind_phrase] + 1) / (
                                unigram_to_count_dict[phrase_list[idx]] + total_words_count)
            else:
                final_prob = 1 * unigram_to_count_dict[phrase_list[0]] / total_words_count
            prob_sen_list.append(final_prob)
        return prob_sen_list

    @time_wrapper
    def generate_backoff_and_interpolation(self, unigram_to_count_dict: dict,
                                           bi_gram_to_count_dict: dict,
                                           inp_list: list):
        prob_sen_list = []
        alpha_1, alpha_2 = 0.3, 0.7
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
    def generate_and_sort_probability_for_each_type(self, inp_list):
        flattened_list = [inp_info for outer_list in inp_list for inp_info in outer_list]
        self.local_logger.info("Probability generation starting for all phrases from conversations")
        self.local_logger.debug("Generating ngrams for the flattened list")
        ngram_list = self.count_ngrams(flattened_list, [1, 2])
        self.local_logger.debug("Generating probabilities for phrases through smoothing techniques")
        additive_theme_prob_list = self.generate_add_one_smoothing(ngram_list[0], ngram_list[1],
                                                                   flattened_list)
        interpol_theme_prob_list = self.generate_backoff_and_interpolation(ngram_list[0], ngram_list[1],
                                                                           flattened_list)
        add_theme_score_list = list(zip(flattened_list, additive_theme_prob_list))
        interpol_theme_score_list = list(zip(flattened_list, interpol_theme_prob_list))
        sorted_add_theme_list = sorted(add_theme_score_list, key=lambda info: info[1], reverse=True)
        sorted_interpol_theme_list = sorted(interpol_theme_score_list, key=lambda info: info[1], reverse=True)
        self.local_logger.debug("Choosing important phrases based on probabilities")
        imp_phrase_list = []
        [imp_phrase_list.append(phrase_info[0]) for phrase_info in sorted_add_theme_list[:10]
         if phrase_info[1] >= self.add_one_smooth_threshold]
        [imp_phrase_list.append(phrase_info[0]) for phrase_info in sorted_interpol_theme_list[:10]
         if phrase_info[1] >= self.interpolation_smooth_threshold]
        return imp_phrase_list
