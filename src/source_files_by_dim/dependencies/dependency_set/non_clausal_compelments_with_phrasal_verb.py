import spacy
from docopt import docopt

from spacy.tokens import DocBin

DEP_PRT_SEPERATOR = "_"

import sys
sys.path.append('../')

#TODO: 100 posts only outputted - {'', 'dative_dobj_prep', 'dative_dobj', 'dobj', 'prep', 'dobj_prep'}
# sets maybe should be more. (8) make sure this isnt a bug
# TODO: output file wit 40000 posts



usage = '''
non_clausal_compelments_with_phrasal_verb.py CLI.
Usage:
    non_clausal_compelments_with_phrasal_verb.py <spacy_file_name> <num_of_posts>
'''

from src.source_files_by_dim.dependencies.abstract_dependency_files import \
    DEP_SET_TYPE, DEP_MODE


from src.source_files_by_dim.dependencies.dependency_set.depedency_set_files import \
    DependencySetFiles


# Phrasal verbs are verbs that appear with a particle.
# In many cases one verb can appear with different particles and each
# verb-particle combination has a different meaning.
# For example:
# show  הראה
# show out   ליווה (אדם) החוצה
# show up   הוֹפִיעַ, צָץ
# show off   הִשְׁוִיץ, הִתְרַבְרֵב, הִצִּיג לְרַאֲוָה
# Because each verb-particle combination has a different meaning
# we can treat each combination as a complex verb, and then collect data
# regarding the argument structure of each complex verb individually.
# This means that for each verb you should check whether it has a PRT dependent.
# If it does then you assign it a separate entry in your data structure.







class NonClausalWithPhrasal(DependencySetFiles):
    def __init__(self, spacy_file_path, groups_mode
                 ):
        # initialize super - dependency_set_files
        DependencySetFiles.__init__(self, spacy_file_path=spacy_file_path,
                                    group_mode= groups_mode,
                                    dep_set_type=DEP_SET_TYPE.NON_CLAUSAL_COMPELEMENTS)
    #TODO: DID NOT ADD METHOD ADD_TOKEN_BY_SINGLE!!!!!!!!
    def add_token_to_dict_by_subset(self, doc_index: int, sent_indx: int,
                                    token: spacy.tokens) -> bool:
        token_children = self.clean_token_children(token)
        # verify that we want to add this token
        if not self.check_if_add_token(token, token_children):
            return False
        # create key for token if not yet in dict keys:
        # if prt is not in the children dependencies, add regularly
        if "prt" not in [child.dep_ for child in token_children]:
            super().add_verb_template_to_dict(token, token_children)
            token_dep_comb = super().arrange_deps(token, token_children)
            super().add_entry_to_dict(token, doc_index, sent_indx,
                                      token_dep_comb, token_children)

        # else add in a special way
        else:
            self.add_verb_template_to_dict(token, token_children)
            token_dep_comb = self.arrange_deps(token, token_children)
            self.add_entry_to_dict(token, doc_index, sent_indx,
                                      token_dep_comb, token_children)

        # arrange dependencies



    """
    override's super's method
    
    each token needs a key in the final dictionary, with it's prt if it exists
    
    if the token.lemma_lower() + prt_word is not yet a key, add as key
    for example:
    verb = go, prt = up -- add entry of "go up" to dictionary
    """

    def add_verb_template_to_dict(self, token: spacy.tokens,
                                  dep_list:list[spacy.tokens]):
        prt_child = self.get_child_with_specidic_dependency(dep_list, "prt")
        token_lemma_with_prt = token.lemma_.lower() + DEP_PRT_SEPERATOR + prt_child.text
        if not token_lemma_with_prt in self.dict_for_csv.keys():
            self.dict_for_csv[token_lemma_with_prt] = {}


    """
    creates dependency set with only dependencies that are not prt
    
    """
    def arrange_deps(self, token: spacy.tokens,
                     token_children: list[spacy.tokens]) -> str:
        if not "prt" in [child.dep_ for child in token_children]:
            raise Exception("irrelevant usage of this method rather than the super's")
        token_children_filtered = filter(lambda child: child.dep_ != "prt", token_children)
        return "_".join(set(sorted([x.dep_ for x in token_children_filtered])))



    def add_entry_to_dict(self, token, doc_index, sent_indx, token_dep_comb,
                          token_children:list[spacy.tokens]):
        # check that prt is not part of the dep_comb
        if "prt" in token_dep_comb:
            raise Exception("arrangement of dependencies left prt in group")

        self.possible_combs.add(token_dep_comb)
        prt_child = self.get_child_with_specidic_dependency(token_children,
                                                            "prt")
        if not prt_child:
            raise Exception("do not have prt as child")
        token_with_prt = token.text + DEP_PRT_SEPERATOR + prt_child.text
        token_lemma_form_with_prt = token.lemma_.lower() +DEP_PRT_SEPERATOR+ prt_child.text

        dict_tuple = (token_with_prt, doc_index, sent_indx,
                      token.sent.text.strip())

        if token_dep_comb in self.dict_for_csv[token_lemma_form_with_prt]:
            self.dict_for_csv[token_lemma_form_with_prt] \
                [token_dep_comb]["counter"] += 1
            self.dict_for_csv[token_lemma_form_with_prt] \
                [token_dep_comb]["instances"].add(dict_tuple)

        else:
            self.dict_for_csv[token_lemma_form_with_prt] \
                [token_dep_comb] = {"instances": set(), "counter": 1}
            self.dict_for_csv[token_lemma_form_with_prt] \
                [token_dep_comb]["instances"].add(dict_tuple)



    """
    this method gets us the first child that has the rquired dependency
    @:param dep_list: list of cleaned children
    @:param dep: required dependency type
    @:return spacy.tokens first child with required dependency, else None
    """
    def get_child_with_specidic_dependency(self, dep_list:list[spacy.tokens],
                                           dep:str)->spacy.tokens:
        for child in dep_list:
            if child.dep_ == dep:
                return child
        return None



if __name__ == '__main__':
    from datetime import datetime

    datetime = datetime.today().strftime('%Y_%m_%d')
    args = docopt(usage)

    file_creator = NonClausalWithPhrasal(spacy_file_path=
                                       args["<spacy_file_name>"],
                                     groups_mode=DEP_MODE.DEP_GROUP)

    csv_path = "dependency_set_from_first_{t}_posts_lg_sents_{n}.csv".format(t=args["<num_of_posts>"],
        n=datetime)
    file_creator.write_dict_to_csv(csv_path)

    counter_path = "dependency_set_from_first_{t}_posts_lg_counter_{n}.csv".format(t=args["<num_of_posts>"],
        n=datetime)
    #TODO: create enum of count and % and other columsn values
    file_creator.write_counter_csv(counter_path, column_set=["COUNT"])

