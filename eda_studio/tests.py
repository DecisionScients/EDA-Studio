# =========================================================================== #
#                                 TESTS                                       #
# =========================================================================== #
# Copyright 2002-2016 The SciPy Developers
# Copyright 2019 John James
#
# The original code from The SciPy Developers was heavily adapted for
# use in EDA Studio by John James.  The original code came with the
# following disclaimer:
#
# Disclaimers:  The function list is obviously incomplete and, worse, the
# functions are not optimized.  All functions have been tested (some more
# so than others), but they are far from bulletproof.  Thus, as with any
# free software, no warranty or guarantee is expressed or implied. :-)  A
# few extra functions that don't appear in the list below can be found by
# interested treasure-hunters.  These functions don't necessarily have
# both list and array versions but were deemed useful.
# such damage. 

"""Exploratory data analysis statistical tests for Python.

This module contains tests parametric and non-parametric tests of:
    - Centrality - Compare means and medians with population or other samples
    - Variances - Two sample and multiple sample variances
    - Association - Between and among quantitative/qualitative variables 
    - Distribution - Compare distributions to theoretical, other, or normal
    - Outliers - Test of outliers on one sample

Centrality
----------
.. autosummary::
   :toctree: generated/

    ttest_one
    ttest_two
    ttest_paired
    anova
    anova_repeated
    wilcoxon
    wilcoxon_signed_rank
    mann_whitney
    kruskal_wallis

Variances
---------
.. autosummary::
   :toctree: generated/

   fishers
   levenes

Association
---------
.. autosummary::
   :toctree: generated/

   chisquare
   pearson
   exact_fisher
   cmh
   spearman
   mantels

Distribution
---------
.. autosummary::
   :toctree: generated/

   kolmogorov_smirnov
   normality

Distribution
---------
.. autosummary::
   :toctree: generated/

   dixon
   grubb   

References
----------
.. [Wasserman:2010:SCC:1965575] Zwillinger, D. and Kokoska, S. (2000). CRC Standard
   Probability and Statistics Tables and Formulae. Chapman & Hall: New
   York. 2000.

"""

# --------------------------------------------------------------------------- #
#                             Centrality Tests                                #
# --------------------------------------------------------------------------- #
def ttest_one(x, mu, axis=0):
    """
    Tests whether a sample comes from a population with a specific mean. 

    The inferential null hypothesis (H:math:'_0'): states that the population
    mean is equal to some value :math:'\mu_0'. The alternative hypothesis (H:math:_a):
    states that the mean does not equal :math:'\mu_0'. The t-statistic 
    standardizes the difference betweeni :math:'\overline{x}' and :math:'\mu_0'.
        \t = \frac{\overline{x}-\mu_0}{\fac{s}{\sqrt{n}}}

        where: :math:'n' is the number of observations in the sample, and
                Degrees of freedom (df) = :math:'n-1'

    Arguments:
        x {array_like} -- Sample observation
        mu {float or array_like} -- Expected value in the null hypothesis.. If 
            array like, then it must have the same shape as :math:'x'.
    
    Keyword Arguments:
        axis {int} -- [description] (default: {0})
    """
    The one-sample t-test is used to determine whether a sample comes from a population with a specific mean. 

