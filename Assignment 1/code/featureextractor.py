from nltk.compat import python_2_unicode_compatible

printed = False

@python_2_unicode_compatible
class FeatureExtractor(object):
    @staticmethod
    def _check_informative(feat, underscore_is_informative=False):
        """
        Check whether a feature is informative
        """

        if feat is None:
            return False

        if feat == '':
            return False

        if not underscore_is_informative and feat == '_':
            return False

        return True

    @staticmethod
    def find_left_right_dependencies(idx, arcs):
        left_most = 1000000
        right_most = -1
        dep_left_most = ''
        dep_right_most = ''
        for (wi, r, wj) in arcs:
            if wi == idx:
                if (wj > wi) and (wj > right_most):
                    right_most = wj
                    dep_right_most = r
                if (wj < wi) and (wj < left_most):
                    left_most = wj
                    dep_left_most = r
        return dep_left_most, dep_right_most

    @staticmethod
    def extract_features(tokens, buffer, stack, arcs):
        """
        This function returns a list of string features for the classifier

        :param tokens: nodes in the dependency graph
        :param stack: partially processed words
        :param buffer: remaining input words
        :param arcs: partially built dependency tree

        :return: list(str)
        """

        """
        Think of some of your own features here! Some standard features are
        described in Table 3.2 on page 31 of Dependency Parsing by Kubler,
        McDonald, and Nivre

        [http://books.google.com/books/about/Dependency_Parsing.html?id=k3iiup7HB9UC]
        """

        result = []


        # Feature Set:
        if stack:
            # STACK[0] Attributes
            stack_idx0 = stack[-1]
            token = tokens[stack_idx0]
            if FeatureExtractor._check_informative(token['word'], True):
                result.append('STK_0_FORM_' + token['word'])

            if FeatureExtractor._check_informative(token['lemma'], True):
                result.append('STK_0_LEMMA_' + token['lemma'])

            if FeatureExtractor._check_informative(token['ctag'], True):
                result.append('STK_0_CTAG_' + token['ctag'])

            if FeatureExtractor._check_informative(token['tag'], True):
                result.append('STK_0_TAG_' + token['tag'])

            if 'feats' in token and FeatureExtractor._check_informative(token['feats']):
                feats = token['feats'].split("|")
                for feat in feats:
                    result.append('STK_0_FEATS_' + feat)

            # Left most, right most dependency of stack[0]
            dep_left_most, dep_right_most = FeatureExtractor.find_left_right_dependencies(stack_idx0, arcs)

            if FeatureExtractor._check_informative(dep_left_most):
                result.append('STK_0_LDEP_' + dep_left_most)
            if FeatureExtractor._check_informative(dep_right_most):
                result.append('STK_0_RDEP_' + dep_right_most)

            # STACK[1] Attributes
                if len(stack) > 1:
                    stack_idx1 = stack[-2]
                    token = tokens[stack_idx1]
                    if FeatureExtractor._check_informative(token['word'], True):
                        result.append('STK_1_FORM_' + token['word'])

                    if FeatureExtractor._check_informative(token['lemma'], True):
                        result.append('STK_1_LEMMA_' + token['lemma'])

                    if FeatureExtractor._check_informative(token['ctag'], True):
                        result.append('STK_1_CTAG_' + token['ctag'])

                    if FeatureExtractor._check_informative(token['tag'], True):
                        result.append('STK_1_TAG_' + token['tag'])

                    if 'feats' in token and FeatureExtractor._check_informative(token['feats']):
                        feats = token['feats'].split("|")
                        for feat in feats:
                            result.append('STK_1_FEATS_' + feat)

        if buffer:
            # BUF[0] Attributes
            buffer_idx0 = buffer[0]
            token = tokens[buffer_idx0]
            if FeatureExtractor._check_informative(token['word'], True):
                result.append('BUF_0_FORM_' + token['word'])

            if FeatureExtractor._check_informative(token['lemma'], True):
                result.append('BUF_0_LEMMA_' + token['lemma'])

            if FeatureExtractor._check_informative(token['ctag'], True):
                result.append('BUF_0_CTAG_' + token['ctag'])

            if FeatureExtractor._check_informative(token['tag'], True):
                result.append('BUF_0_TAG_' + token['tag'])

            if 'feats' in token and FeatureExtractor._check_informative(token['feats']):
                feats = token['feats'].split("|")
                for feat in feats:
                    result.append('BUF_0_FEATS_' + feat)

            # Left most, right most dependency of BUF[0]
            dep_left_most, dep_right_most = FeatureExtractor.find_left_right_dependencies(buffer_idx0, arcs)

            if FeatureExtractor._check_informative(dep_left_most):
                result.append('BUF_0_LDEP_' + dep_left_most)
            if FeatureExtractor._check_informative(dep_right_most):
                result.append('BUF_0_RDEP_' + dep_right_most)

            # BUF[1] Attributes
            if len(buffer) > 1:
                buffer_idx1 = buffer[1]
                token = tokens[buffer_idx1]
                if FeatureExtractor._check_informative(token['word'], True):
                    result.append('BUF_1_FORM_' + token['word'])

                if FeatureExtractor._check_informative(token['lemma'], True):
                    result.append('BUF_1_LEMMA_' + token['lemma'])

                if FeatureExtractor._check_informative(token['ctag'], True):
                    result.append('BUF_1_CTAG_' + token['ctag'])

                if FeatureExtractor._check_informative(token['tag'], True):
                    result.append('BUF_1_TAG_' + token['tag'])

                if 'feats' in token and FeatureExtractor._check_informative(token['feats']):
                    feats = token['feats'].split("|")
                    for feat in feats:
                        result.append('BUF_1_FEATS_' + feat)

            # BUF[2] Attributes
            if len(buffer) > 2:
                buffer_idx2 = buffer[2]
                token = tokens[buffer_idx2]
                if FeatureExtractor._check_informative(token['word'], True):
                    result.append('BUF_2_FORM_' + token['word'])

                if FeatureExtractor._check_informative(token['lemma'], True):
                    result.append('BUF_2_LEMMA_' + token['lemma'])

                if FeatureExtractor._check_informative(token['ctag'], True):
                    result.append('BUF_2_CTAG_' + token['ctag'])

                if FeatureExtractor._check_informative(token['tag'], True):
                    result.append('BUF_2_TAG_' + token['tag'])

                if 'feats' in token and FeatureExtractor._check_informative(token['feats']):
                    feats = token['feats'].split("|")
                    for feat in feats:
                        result.append('BUF_2_FEATS_' + feat)

        return result
