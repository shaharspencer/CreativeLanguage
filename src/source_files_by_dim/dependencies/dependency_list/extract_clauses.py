import re

import spacy.tokens
from spacy.tokens import DocBin
from src.generate_and_test_spacy.processors.processor import Processor
import csv

# TODO perhaps remove double spaces
NON_RELEVANT_DEPS = {"advcl", "relcl", "advmod", "acl", "punct"}
RELEVANT_DEPS = {"nsubj", "nsubjpass", "ccomp", "csubj",
                 "xcomp", "prt", "dobj", "prep", "dative", "pobj"}

WH_WORDS = {"what", "which", "whatever", "who", "whom"
    , "whateva", "whoever", "whichever", "when", "whenever", "how"}


def create_field_dict(fieldnames):
    dic = {}
    for fieldname in fieldnames:
        dic[fieldname] = fieldname
    return dic


def is_valid_token(token):
    return token.text.lower() not in WH_WORDS


class ExtractClauses:
    def __init__(self, spacy_path: str):
        self.nlp = Processor(to_conllu=False, use_ensemble_tagger=True,
                             to_process=False).get_nlp()
        # deps = self.nlp.get_pipe("parser").labels
        # d = self.nlp.meta["sources"]
        # for dep in deps:
        #     spacy.explain(dep)
        self.doc_bin = DocBin().from_disk(spacy_path)

    def trial(self):
        csv_file_path = 'eat_clause_example_yay.csv'
        self.get_dobj(csv_file_path)

    # def get_clause(self, token):
    #     for doc in self.doc_bin.get_docs(self.nlp.vocab):
    #         for token in doc:
    #             if token.pos_ == "VERB" and token.lemma_.lower() == "watch":
    #                 print(f"sentence number {counter}")
    #                 counter += 1
    #                 if counter == 2000:
    #                     return
    #                 self.write_token_row_clause(writer, token=token)
    #
    #
    # def write_token_row_clause(self, writer, token: spacy.tokens):
    #     token_subtree = self.extract_subtree(token=token,
    #                                          orig_token_id=token.i)
    #     token_subtree_sent = self.get_sent(token_subtree)

    def get_dobj(self, csv_file_path):
        with open(csv_file_path, 'w', encoding='utf-8',
                  newline='') as csv_file:
            # create a CSV writer object
            fieldnames = ["lemma (V)", "sentence",
                          "dobj",
                          "dobj index"]
            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            csv_writer.writerow(create_field_dict(fieldnames=fieldnames))
            self.iterate_over_docbin_dobj(csv_writer)

    def iterate_over_docbin_dobj(self, writer):
        counter = 0
        for doc in self.doc_bin.get_docs(self.nlp.vocab):
            for token in doc:
                if token.pos_ == "VERB" and token.lemma_.lower() == "eat":
                    print(f"sentence number {counter}")
                    counter += 1
                    if counter == 2000:
                        return
                    self.write_token_row_dobj(writer, token=token)

    def write_token_row_dobj(self, writer, token: spacy.tokens):
        dobj_lst = self.get_all_dobjs(token)
        for dobj_tuple in dobj_lst:
            n_dict = {"lemma (V)": token.lemma_.lower(),
                      "sentence": token.doc.text,
                      "dobj": dobj_tuple[0],
                      "dobj index": dobj_tuple[1]}
            writer.writerow(n_dict)


    def get_all_dobjs(self, token: spacy.tokens) -> list[tuple[str, int]]:
        dobj_lst = []
        for child in token.children:
            if child.dep_ == "dobj" and child.pos_ == "NOUN":
                dobj_lst.append((child.text, child.i))

        return dobj_lst


    def extract_subtree(self, token: spacy.tokens,
                        orig_token_id: int) -> list[
                                                    spacy.tokens] | None:
        if (token.dep_ in NON_RELEVANT_DEPS or not is_valid_token(token)) \
                and not token.i == orig_token_id:
            return None

        subtree = [token]
        for child in token.children:
            child_subtree = self.extract_subtree(token=child,
                                                 orig_token_id=orig_token_id)
            if child_subtree:
                subtree.extend(child_subtree)

        return subtree

    def get_sent(self, token_subtree: list[spacy.tokens]):
        token_subtree.sort(key=lambda k: k.i)
        token_subtree_sorted_str = [token.text.lemma_.lower() for token in token_subtree]
        token_subtree_sent = " ".join(token_subtree_sorted_str)
        return token_subtree_sent


    # def extract_all_relevant_dep_clauses(self, token: spacy.tokens,
    #                                      subtree: list[spacy.tokens]):
    #     min_token, max_token = self.find_clause_boundaries(subtree=subtree)
    #     claused_sent = token.doc[min_token.i: max_token.i + 1]. \
    #         text
    #     return claused_sent

    # edge case: no children, word is not included in
    # def find_clause_boundaries(self, subtree) -> \
    #         tuple[spacy.tokens, spacy.tokens]:
    #     try:
    #         min_clause_token = min(subtree, key=lambda child: child.i)
    #         max_clause_token = max(subtree, key=lambda child: child.i)
    #         return min_clause_token, max_clause_token
    #     except Exception:
    #         raise Exception("problem :)")


if __name__ == '__main__':
    obj = ExtractClauses(
        spacy_path=r"C:\Users\User\OneDrive\Documents\CreativeLanguageOutputFiles\training_data\spacy_data\GPU_7_30_2023_data_from_first_0_lg_model_spacy_3.5.5.spacy")
    obj.trial()
