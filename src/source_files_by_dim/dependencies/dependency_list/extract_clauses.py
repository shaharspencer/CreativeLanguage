import spacy.tokens
from spacy.tokens import DocBin
from src.generate_and_test_spacy.processors.processor import Processor
import csv


#TODO perhaps remove double spaces
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
        csv_file_path = 'watch_clause_example_yay.csv'
        with open(csv_file_path, 'w', encoding='utf-8', newline='') as csv_file:
            # Create a CSV writer object
            fieldnames = ["lemma (V)", "sentence", "clause"]
                #, "deps list"]
            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            csv_writer.writerow(create_field_dict(fieldnames=fieldnames))
            self.iterate_over_docbin(csv_writer)


    def iterate_over_docbin(self, writer):
        counter = 0
        sent = "The man who loves to fly watched a movie."
        nlp_sent = self.nlp(sent)
        for token in nlp_sent:
            subtree = self.extract_subtree_two(token, orig_token_index=token.i)
        y = "hi"
        # for doc in self.doc_bin.get_docs(self.nlp.vocab):
        #     for token in doc:
        #         if token.pos_ == "VERB" and token.lemma_.lower() == "watch":
        #             print(f"sentence number {counter}")
        #             counter += 1
        #             if counter == 500:
        #                 return
        #             if token.doc == "Then i got to thinking (well, it came to me while watching a kia ad) that morning is everywhere in korea.":
        #                 y = "hi"
        #             row = self.create_row_dict(token=token)
        #             writer.writerow(row)

    def create_row_dict(self, token: spacy.tokens)->dict:
        token_subtree = self.extract_subtree(token=token)
        if not token_subtree:
            x = 0
        all_relevant_dep_sent = self.extract_all_relevant_dep_clauses(
            token, token_subtree)
        n_dict = {"lemma (V)": token.lemma_.lower(),
                  "sentence": token.doc.text,
                  "clause": all_relevant_dep_sent,
                  # "deps list": dep_list
                  }
        return n_dict

    def extract_subtree_two(self,token, orig_token_index):
        if (token.dep_ in NON_RELEVANT_DEPS or not is_valid_token(token)) and token.i != orig_token_index:
            return None

        min_index = token.i
        max_index = token.i
        included_tokens = set()

        def process_subtree(current_token):
            nonlocal min_index, max_index
            if not is_valid_token(current_token):
                return

            included_tokens.add(current_token)
            min_index = min(min_index, current_token.i)
            max_index = max(max_index, current_token.i)

            for child in current_token.children:
                if child.i > max_index or child.i < min_index:
                    continue
                if child.dep_ not in NON_RELEVANT_DEPS:
                    process_subtree(child)

        process_subtree(token)

        relevant_subtree = []
        for token in included_tokens:
            if token.dep_ not in NON_RELEVANT_DEPS:
                relevant_subtree.append(token)

        return relevant_subtree

    def extract_subtree(self, token: spacy.tokens) -> list[
                                                          spacy.tokens] | None:
        if token.dep_ in NON_RELEVANT_DEPS or not is_valid_token(token):
            return None

        subtree = [token]
        for child in token.children:
            child_subtree = self.extract_subtree(child)
            if child_subtree:
                subtree.extend(child_subtree)

        return subtree

    def extract_all_relevant_dep_clauses(self, token: spacy.tokens,
                                         subtree: list[spacy.tokens]):
        min_token, max_token = self.find_clause_boundaries(subtree=subtree)
        claused_sent = token.doc[min_token.i: max_token.i + 1]. \
            text
        return claused_sent

    # edge case: no children, word is not included in
    def find_clause_boundaries(self, subtree) -> \
            tuple[spacy.tokens, spacy.tokens]:
        try:
            min_clause_token = min(subtree, key=lambda child: child.i)
            max_clause_token = max(subtree, key=lambda child: child.i)
            return min_clause_token, max_clause_token
        except Exception:
            raise Exception("problem :)")




if __name__ == '__main__':
    obj = ExtractClauses(
        spacy_path=r"C:\Users\User\OneDrive\Documents\CreativeLanguageOutputFiles\training_data\spacy_data\GPU_7_30_2023_data_from_first_0_lg_model_spacy_3.5.5.spacy")
    obj.trial()
