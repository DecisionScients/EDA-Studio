# =========================================================================== #
#                                UNIVARIATE                                   #
# =========================================================================== #
# univariate.py 
#
# BSD License
#
# Copyright (c) 2019, John James
# All rights reserved.
#region 
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice, this
#   list of conditions and the following disclaimer in the documentation and/or
#   other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE.
#endregion 

"""Module for conducting univariate exploratory data analysis in Python

This module contains functions, a class and its methods for conducting
univariate exploratory data analysis on a pandas series, dataframe, or
numpy arrays.

"""
import numpy as np
import pandas as pd

class Univariate(object):
    """Conducts, stores and reports univariate exploratory data analysis.
    
    Class performs graphical and non-graphical exploratory data analysis
    on quantitative (continuous, discrete) and qualitative (nominal, 
    ordinal, binomial) data.

    Univariate analyses are only conducted in pandas.DataFrame and
    pandas.Series objects.

    """

    def __init__(self, x):
        """Initializes the class with the object to be analyzed.

        Arguments:
        ----------
            x (pandas DataFrame): Qualitative and or quantitative data to 
                be analyzed.

        """
        self.x = x

    def _describe_qual(self, df, quantiles, include, exclude):
        """Performs the describe function for qualitative variables

        This method performs the qualitative analysis on categorical 
        variables. It includes counts and frequencies of values, 
        as well as the value and frequency of the most frequently occurring
        element.

        Arguments:
        ----------
        df (DataFrame): Qualitative data to be analyzed
        **kwargs
            Arbitrary keyword arguments as follows:
            
            - ``quantiles``: (array-like): list-like of numbers, optional
                The quantiles to include in the output. The values
                should be between 0 and 1.  The default is 
                ``[.25, .5, .75]``, which returns the 25th, 50th, and
                75th quantiles.
            - ``include``: (str or array-like): A string or an array-like specifying
                the name(s) of the column(s) or the names of the data types
                to include in the analysis. Valid data types include:
                
                * numpy.object 
                * numpy.bool

                The default value is None, which means that all categorical
                columns will be included in the analysis.

            - ``exclude``: (str or array-like): A string or an array-like 
                specifying the name(s) of the column(s) or data types to 
                exclude from the analysis. See the list of valid data types
                above. The default value is None, which means that nothing 
                will be excluded. 

        Returns:
        --------
            DataFrame: Contains count and frequency data. 

        """
        result = pd.DataFrame()
        cols = df.columns
        for col in cols:
            d = pd.DataFrame(df[col].describe(percentile=quantiles, 
                                              include=include,
                                              exclude=exclude))
            d = d.T
            d['missing'] = df[col].isna().sum()
            result = result.append(d)
        return(result)
        



    def describe(self, quantiles=None, include=None, exclude=None):
        """Descriptive statistics built upon the pandas.describe class.

        This method provides a summary and descriptive statistics for both
        quantitative and qualitative data. 
        
        Quantitative Analysis (Numeric Data)
        ------------------------------------    
        A quantitative analysis of the numeric variables includes 
        measures of central tendency (mean,mode, median), distribution 
        (min, quartiles, max), variance (standard deviation), and shape (normality, 
        skew, kurtosis). A  a Shapiro-Wilk test for normality is conducted.l The 
        Shapiro-Wilk test evaluates the null hypothesis that the data was drawn
        from a normal distribution. The function also indicates whether a 
        distribution is kurtotic or skewed according to the following rubric:    

        ::

            A distribution is skewed if its skewness is greater than two standard
            errors, negatively or positively from the mean. Similarly, a 
            distribution is kurtotic if its kurtosis measurement is beyond
            two standard errors from the mean.

        Qualitative Analysis (Categorical Data)
        ---------------------------------------
        Finally, a qualitative analysis of the categorical variables will 
        include counts and frequencies of values, as well as the value and 
        frequency of the most frequently occurring element.
        
        Arguments:
        ----------
            quantiles (array-like): list-like of numbers, optional
                The quantiles to include in the output. The values
                should be between 0 and 1.  The default is 
                ``[.25, .5, .75]``, which returns the 25th, 50th, and
                75th quantiles.

            include (str or array-like): A string or an array-like specifying
                the name(s) of the column(s) or the names of the data types
                to include in the analysis. Valid data types include:
                
                * number
                * numpy.number   
                * numpy.object 
                * numpy.bool

                The default value is None, which means that all columns
                will be included in the analysis.

            exclude (str or array-like): A string or an array-like specifying
                the name(s) of the column(s) or data types to exclude from the 
                analysis. See the list of valid data types above. The default 
                value is None, which means that nothing will be excluded. 

        Returns:
        --------
            Dictionary: Containing two DataFrames:  
                quantitative (DataFrame): quantitative analysis of numerics
                qualitative (DataFrame): qualitative analysis of categoricals

        Notes:
        ------
        If there are no numeric columns in the input, the dictionary will not
        contain a quantitative member. Similarly, if no categorical data is
        extant, the qualitative analysis will be omitted from the result. 
        If object contains data of mixed types, it will be treated
        as a qualitative analysis.

        Examples:
        ---------
        Describe a numeric ``Series``.

        >>> s = pd.Series([3, 42, 8, 5, 22, 13,79, 43, 2, 105, 6, 6])
        >>> d = describe(s)
        >>> d['quantitative'].T    # Transposed for viewing purposes
        count     12.000000
        mean      27.833333
        std       33.587967
        min        2.000000
        25%        5.750000
        50%       10.500000
        75%       42.250000
        max      105.000000
        kurtosis 0.4411928295956846
        kurtotic positive kurtosis
        skew     1.3111792977843535   
        skewed   positive skew

        >>> d['qualitative'] 
        None


        Describe a mixed datatype ``DataFrame``.

        >>> df = pd.DataFrame({'object': ['a','c', 'united', 'tofu', 'loathing',
                                        'green', 'b', 'e', 'q', 'green', 'rover'
                                        'slack'],
        ...                    'numeric' : [3, 42, 8, 5, 22, 13,79, 43, 2, 105, 6, 6]})
        >>> d = describe(df)
        >>> d['summary']
        dtype:      pandas.core.frame.DataFrame
        rows:       12
        columns:    2
        column_names: ['object', 'numeric']
        quantitative 1
        qualitative  1
        NaN          0
        null         0

        >>> d['quantitative'].T  # Transposed for ease of presentation
                numeric
        count     12.000000
        mean      27.833333
        std       33.587967
        min        2.000000
        25%        5.750000
        50%       10.500000
        75%       42.250000
        max      105.000000
        kurtosis 0.4411928295956846
        kurtotic positive kurtosis
        skew     1.3111792977843535   
        skewed   positive skew

        >>> d['qualitative'].T # Transposed for ease of presentation
                object
        count   12
        unique  11
        top     green
        freq    2

        """

        # Initialize output
        result = {}
        result['quantitative'] = None
        result['qualitative'] = None

        # Validation
        if 'pandas' not in str(type(self.x)):
            raise Exception("Can only describe pandas Series or DataFrames.")

        if 'Series' in str(type(self.x)):
            if sum(np.isreal(self.x)) == len(self.x):
                result['quantitative'] = self._describe_quant(self.x.to_frame(),
                                                              quantiles=quantiles, 
                                                              include=include,
                                                              exclude=exclude)
            else:
                result['qualitative'] = self._describe_qual(self.x.to_frame(),
                                                            quantiles=quantiles, 
                                                            include=include,
                                                            exclude=exclude)

        else:
            qual = self.x.select_dtypes([np.bool, np.object])

            quant = self.x.select_dtypes([np.number])
            cols = self.x.columns




