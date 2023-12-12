import json
from collections import Counter

from conllu import parse, SentenceList



def get_frequency_band_from_list(pos_word_list: list[str], num_bands = 99)->dict:
    pos_counter = Counter(pos_word_list)
    total_pos = sum(pos_counter.values())
    sorted_pos = sorted(pos_counter.items(), key=lambda x: x[1], reverse=True)

    frequency_bands_for_pos = {}
    band_size = total_pos // num_bands
    cumulative_frequency = 0
    current_band = 1

    for word, count in sorted_pos:
        cumulative_frequency += count
        if cumulative_frequency > band_size * current_band:
            current_band += 1

        frequency_bands_for_pos[word] = current_band

    return frequency_bands_for_pos


def analyze_frequency_bands(open_conllu_files):
    nouns, verbs, adjs, adp, adv, det, propn = [], [],[],[],[],[],[]

    for file in open_conllu_files:
        sent_counter = 1
        for sentence in file:
            for token in sentence:
                if token["upos"] == "NOUN":
                    nouns.extend([token["lemma"]])
                if token["upos"] == "VERB":
                    verbs.extend([token["lemma"]])
                if token["upos"] == "ADJ":
                    adjs.extend([token["lemma"]])
                if token["upos"] == "ADP":
                    adp.extend([token["lemma"]])
                if token["upos"] == "ADV":
                    adv.extend([token["lemma"]])
                if token["upos"] == "DET":
                    det.extend([token["lemma"]])
                if token["upos"] == "PROPN":
                    propn.extend([token["lemma"]])


            sent_counter += 1


    frequency_bands = {}
    for pos, pos_list in zip(["NOUN", "VERB", "ADJ", "ADP", "ADV", "DET", "PROPN" ], [nouns, verbs, adjs, adp, adv, det, propn]):
        frequency_bands[pos] = get_frequency_band_from_list(pos_list)

    return frequency_bands

def open_conllu(conllu_file_path)-> SentenceList:
    with open(conllu_file_path, 'r', encoding='utf-8') as conllu_file:
        conllu_content = parse(conllu_file.read())
    return conllu_content
if __name__ == '__main__':

    open_conllu_file_test = open_conllu(r"C:\Users\User\PycharmProjects\CreativeLanguage\src\masking_subproject\files\raw_data\en_ewt-ud-test.conllu")
    open_conllu_file_val = open_conllu(r"C:\Users\User\PycharmProjects\CreativeLanguage\src\masking_subproject\files\raw_data\en_ewt-ud-dev.conllu")
    open_conllu_file_train = open_conllu(r"C:\Users\User\PycharmProjects\CreativeLanguage\src\masking_subproject\files\raw_data\en_ewt-ud-train.conllu")
    freq_bands = analyze_frequency_bands(open_conllu_files=[open_conllu_file_test,
                                                            open_conllu_file_val,
                                                            open_conllu_file_train])
    json_file_path = "freq_bands_100.json"

    # Dump the dictionary to a JSON file
    with open(json_file_path, 'w') as json_file:
        json.dump(freq_bands, json_file, indent=4)

