# Generalization of RNN__Transform & CNN__Transform
#
# The classes NNTrainTransform and NNPredictTransform are replacements of
#
#   RNNTrainTransform, RNNPredictTransform
#   CNNTrainTransform, CNNPredictTransform
#
# The classes RNN__Transform and CNN__Transform differ only for this reason:
#
#                        1  2                3
#   RNN__Tranform   ->  (n, sequence_length, data_size)
#   CNN__Tranform   ->  (n, channel_size,    channel_length)
#
# that is, there is only a 'swap' between columns 2 and 3.
# Now, it is possible to 'normalize' the CNN transformers applying a 'swap'
# on the CNN channels:
#
#   CNN'_Transform  ->  (n, channel_length, channel_size)
#
# in such way to have the same 'layout' of RNN.
#

from sktimex import LagsTrainTransform, LagsPredictTransform


# ---------------------------------------------------------------------------
# NNTrainTransform
# NNPredictTransform
# ---------------------------------------------------------------------------

class NNTrainTransform(LagsTrainTransform):
    def __init__(self, xlags=None, ylags=None, tlags=None, ulags=None, yprev=False, ytrain=False, flatten=False):
        """
        :param xlags:
        :param ylags:
        :param tlags:
        :param ytrain:  if to return y used in train (yx)
        :param yprev:   if to return y[t-1]
        :param flatten: if to return 2D arrays (n, len(tlags)*data_size)
                        or 3D arrays (n, len(tlags), data_size)
        """
        super().__init__(
            xlags=xlags, ylags=ylags, tlags=tlags, ulags=ulags, flatten=flatten,
        )
        self.yprev = yprev
        self.ytrain = ytrain
    # end

    def predict_transform(self) -> "NNPredictTransform":
        return NNPredictTransform(
            xlags=self.xlags, ylags=self.ylags, tlags=self.tlags, ulags=self.ulags, flatten=self.flatten
        )
# end


class NNPredictTransform(LagsPredictTransform):
    def __init__(self, xlags=None, ylags=None, tlags=None, ulags=None, flatten=False):
        super().__init__(xlags=xlags, ylags=ylags, tlags=tlags, ulags=ulags, flatten=flatten)

# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
