import spacy
from docopt import docopt

from spacy.tokens import DocBin

DEP_PRT_SEPERATOR = "_"

import sys
sys.path.append('../')

#TODO: 100 posts only outputted - {'', 'dative_dobj_prep', 'dative_dobj', 'dobj', 'prep', 'dobj_prep'}
# sets maybe should be more. (8) make sure this isnt a bug
# TODO: output file wit 40000 posts

# <dep_set_type>
# NON_CLAUSAL: DEP_SET_TYPE.NON_CLAUSAL_COMPELEMENTS
# COMPLETE: DEP_SET_TYPE.COMPLETE_COMPELEMENTS


# <set_or_single>
# "SET": DEP_MODE.DEP_GROUP
# "SINGLE": DEP_MODE.SINGLE_DEP


usage = '''
phrasal_verbs.py CLI.
Usage:
    phrasal_verbs.py <spacy_file_name> <num_of_posts> <dep_set_type> <set_or_single>
'''

from src.source_files_by_dim.dependencies.abstract_dependency_files import \
    DEP_SET_TYPE, DEP_MODE


from src.source_files_by_dim.dependencies.dependency_set.depedency_set_files import \
    DependencySetFiles
from src.source_files_by_dim.dependencies.abstract_dependency_files import DependencyDimensionFiles


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



class PhrasalVerbs(DependencySetFiles):
    def __init__(self, spacy_file_path, groups_mode, dep_set_type
                 ):
        # initialize super - dependency_set_files
        DependencySetFiles.__init__(self, spacy_file_path=spacy_file_path,
                                    group_mode= groups_mode,
                                    dep_set_type=dep_set_type)

    """
    this method adds dependencies to dictionary one by one. 
    if we have a prep dependency we add the prep word itself. 
    """
    def add_token_to_dict_by_single_dep(self, doc_index: int, sent_indx: int,
                                    token: spacy.tokens) -> bool:
        # get list of children w/o "punct" and such
        token_children = self.clean_token_children(token)

        # verify that we want to add this token
        if not self.check_if_add_token(token, token_children):
            return False


        # if prt is in children dependencies, turn token into token + prt
        prt_child = self.get_child_with_specific_dependency(token_children,
                                                             "prt")


        if prt_child:
            self.handle_prt_child_add_single_dep(doc_index=doc_index,
                                                 sent_indx=sent_indx,
                                                 token=token,
                                                 token_children=token_children,
                                                 )


        else:
            self.handle_not_prt_child_add_single_dep(doc_index=doc_index,
                                                 sent_indx=sent_indx,
                                                 token=token,
                                                 token_children=token_children
                                                 )

    """
    the add_token_to_dict_by_single_dep method uses this function incase 
    the token children contains a prt dependency. we then want to 
    add entry to dictionary of word with the prt dependency and 
    add each dependency one by one. 
    
    """
    def handle_prt_child_add_single_dep(self, doc_index: int, sent_indx: int,
                                    token: spacy.tokens, token_children):
        self.add_verb_template_to_dict(token, token_children)

        if not token_children:
            self.add_entry_to_dict(token, doc_index, sent_indx,
                                   "",
                                   token_children)
            return True
            # else if not empty dep list
        for dep in token_children:
            if dep.dep_ == "prep":
                self.add_entry_to_dict(token, doc_index, sent_indx,
                                       dep.text,
                                       token_children)
            # prt becomes part of the verb
            elif dep.dep_ == "prt":
                pass

            else:
                self.add_entry_to_dict(token, doc_index, sent_indx,
                                       dep.dep_,
                                       token_children)

    """
    the add_token_to_dict_by_single_dep method uses this function incase 
    the token children does NOT contain a prt dependency. we then want to 
    add entry to dictionary of word by itself and then
    add each dependency one by one. 

    """

    def handle_not_prt_child_add_single_dep(self, doc_index: int, sent_indx: int,
                                        token: spacy.tokens, token_children,
                                     ):
        super().add_verb_template_to_dict(token, token_children)
        # if empty dep list
        if not token_children:
            super().add_entry_to_dict(token, doc_index, sent_indx,
                                      "",
                                      token_children)
            return True
        # else if not empty dep list
        for dep in token_children:
            if dep.dep_ == "prep":
                super().add_entry_to_dict(token, doc_index, sent_indx,
                                          dep.lemma_.lower(),
                                          token_children)
            else:
                super().add_entry_to_dict(token, doc_index, sent_indx,
                                          dep.dep_,
                                          token_children)
        return True





    #TODO: note - subset does not have the actual prep word like single does.
    """
    this method adds dependencies to dictionary as 
    an entire set of dependencies. 
    """
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



    """
    override's super's method
    
    each token needs a key in the final dictionary, with it's prt if it exists
    
    if the token.lemma_lower() + prt_word is not yet a key, add as key
    for example:
    verb = go, prt = up -- add entry of "go up" to dictionary
    """

    def add_verb_template_to_dict(self, token: spacy.tokens,
                                  dep_list:list[spacy.tokens]):
        prt_child = self.get_child_with_specific_dependency(dep_list, "prt")
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



    def add_entry_to_dict(self, token: str, doc_index, sent_indx, token_dep_comb,
                          token_children:list[spacy.tokens]):
        # check that prt is not part of the dep_comb
        if "prt" in token_dep_comb:
            raise Exception("arrangement of dependencies left prt in group")

        self.possible_combs.add(token_dep_comb)
        prt_child = self.get_child_with_specific_dependency(token_children,
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
    def get_child_with_specific_dependency(self, dep_list:list[spacy.tokens],
                                           dep:str)->spacy.tokens:
        for child in dep_list:
            if child.dep_ == dep:
                return child
        return None


"""
    send the argument corresponding to the dependency set type here
    to get the type we want:
    non clausal compelements: prt, dobj, prep, dative
    complete compelements: ccomp, xcomp, prt, dobj, prep, dative
"""

def get_dep_set_type(arg) -> DEP_SET_TYPE:

    if arg == "NON_CLAUSAL":
        return DEP_SET_TYPE.NON_CLAUSAL_COMPELEMENTS

    elif arg == "COMPLETE":
        return DEP_SET_TYPE.COMPLETE_COMPELEMENTS

    else:
        raise Exception("dep set type is not legal")

"""
    send the argument corresponding to the group type here 
    to get the type we want:
    single dependency - add single denpdency one by one, for prep add the actual word
    group dependencies - add entire set of dependencies

"""

def get_group_type(param) -> DEP_MODE:
    if param == "SET":
        return DEP_MODE.DEP_GROUP
    elif param == "SINGLE":
        return DEP_MODE.SINGLE_DEP
    else:
        raise Exception("dep set type is not legal")




if __name__ == '__main__':
    from datetime import datetime

    datetime = datetime.today().strftime('%Y_%m_%d')
    args = docopt(usage)

    dep_set_type = get_dep_set_type(args["<dep_set_type>"])

    group_type = get_group_type(args["<set_or_single>"])


    file_creator = PhrasalVerbs(spacy_file_path=
                                       args["<spacy_file_name>"],
                                groups_mode=group_type,
                                dep_set_type=dep_set_type)

    csv_path = \
        "dependency_set_from_first_{t}_posts_lg_sents_{n}_{x}_{mode}.csv".format(
            t=args["<num_of_posts>"], x=args["<dep_set_type>"], mode = args["<set_or_single>"],
        n=datetime)
    file_creator.write_dict_to_csv(csv_path)

    counter_path = \
        "dependency_set_from_first_{t}_posts_lg_counter_{n}_{x}_{mode}.csv".format(
            t=args["<num_of_posts>"],
                      x=
                      args[
                          "<dep_set_type>"],
                      mode=
                      args[
                          "<set_or_single>"],
                      n=datetime)
    #TODO: create enum of count and % and other columsn values
    file_creator.write_counter_csv(counter_path, column_set=["COUNT"])

