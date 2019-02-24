# =========================================================================== #
#                                UNIVARIATE                                   #
# =========================================================================== #
# univariate.py 
#
# BSD License
#
# Copyright (c) 2019, John James
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice, this
#   list of conditions and the following disclaimer in the documentation and/or
#   other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.
#
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

"""Module for conducting univariate exploratory data analysis in Python

This module contains functions, a class and its methods for conducting
univariate exploratory data analysis on a pandas series, dataframe, or
numpy arrays.

"""
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew, sem, shapiro, mode
from statistics import mean
import warnings
warnings.filterwarnings("ignore")

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
            x (pandas Series or DataFrame): Qualitative and or quantitative data to 
                be analyzed.

        """
        self.x = x
        self.description = None

    def _describe_qual(self, df, include='all', exclude=None):
        """Performs the describe function for qualitative variables

        This method performs the qualitative analysis on categorical 
        variables. It includes counts and frequencies of values, 
        as well as the value and frequency of the most frequently occurring
        element.

        Arguments:
        ----------
        df (DataFrame): Qualitative data to be analyzed
        
        include: (str or array-like): A string or an array-like specifying
        the name(s) of the column(s) or the names of the data types
        to include in the analysis. Valid data types include:
        
        * numpy.object 
        * numpy.bool

        The default value is None, which means that all categorical
        columns will be included in the analysis.

        exclude: (str or array-like): A string or an array-like 
        specifying the name(s) of the column(s) or data types to 
        exclude from the analysis. See the list of valid data types
        above. The default value is None, which means that nothing 
        will be excluded. 

        Returns:
        --------
            DataFrame: Contains count and frequency data. 

        """
        c = df.dtypes[df.dtypes=='object'].index.values
        desc = df[c].describe(include=include, exclude=exclude)
        
        obs = pd.DataFrame({'n':np.repeat(df[c].shape[0],len(c))},index=c)
        unique = pd.DataFrame({'Unique':df[c].apply(lambda x:len(x.unique()),axis=0)})
        missing = pd.DataFrame({'Missing':df[c].isnull().sum()})
        pct_missing = pd.DataFrame({'% Missing':df[c].isnull().sum()/df[c].shape[0]*100})                
        top = pd.DataFrame({'Top': desc.loc['top', :]})
        top_freq = pd.DataFrame({'Top Freq': desc.loc['freq', :]})
        dq = pd.concat([obs, unique, missing, pct_missing, top, top_freq], axis=1)
        return(dq)

    def _skew(self, y):
        """Interprets skew of a distribution

        The following rule of thumb is used to interpret skewness:

        * If skewness is less than −1 or greater than +1, the distribution is highly skewed.  
        * If skewness is between −1 and −½ or between +½ and +1, the distribution is moderately skewed.  
        * If skewness is between −½ and +½, the distribution is approximately symmetric.   

        Arguments:
        ----------
        y (float): Measure of skewness

        Returns:
        --------
        String: Contains the interpretation. 

        """
        skewed = np.where(y < -1, 'high negative skew',
                     np.where(y <= -0.5, 'moderate negative skew',
                     np.where(y < 0.5, 'symmetric',
                     np.where(y < 1, 'moderate positive skew',
                     'high positive skew'))))
        return(skewed)

    def _kurtosis(self, g):
        """Interprets kurtosis of a distribution

        The following rule of thumb is used to interpret kurtosis:

        * if kurtosis < -3: platykurtic
        * if kurtosis < -2: likely platykurtic
        * if kurtosis is between -2 and +2, inconclusive
        * if kurtosis > 2: likely leptokurtic
        * if kurtosis > 3: leptokurtic
        
        Arguments:
        ----------
        g (float): Measure of kurtosis

        Returns:
        --------
        String: Contains the interpretation. 

        """
        kurtosic = np.where(g < -3, 'platykurtic', 
                    np.where(g < -2, 'likely platykurtic',
                    np.where(g < 2, 'inconclusive',
                    np.where(g < 3, 'likely leptokurtic',
                    'leptokurtic'))))

        return(kurtosic)        

    def _describe_quant(self, df, quantiles=None, include='all', exclude=None, sig=0.05):
        """Computes descriptive statistics for quantitative variables

        Built upon the pandas.DataFrame.describe class, this method provides
        descriptive statistics for a quantitative variable. The statistics include:

            * number of missing values
            * min, max, mean, mode and median   
            * quantiles, defaults are [0.25, .50, 0.75]    
            * standard deviation, standard error    
            * kurtosis, skewness and normality

        Arguments:
        ----------
            df(DataFrame): One column DataFrame to be analyzed
            quantiles(float): The quantiles to be returned    
            include(list-like): List of columns or data types to be included
            exclude(list-like): List of columns or data types to be excluded
            sig(float): The significance level for the normality test

        Returns:
        --------
        DataFrame: Containing descriptive statistics.

        """
 
        d=df.dtypes[df.dtypes!='object'].index.values
        df[d]=df[d].astype('float64')

        obs = pd.DataFrame({'n':np.repeat(df[d].shape[0],len(d))},index=d)
        unique = pd.DataFrame({'Unique':df[d].apply(lambda x:len(x.unique()),axis=0)})
        missing = pd.DataFrame({'Missing':df[d].isnull().sum()})
        pct_missing = pd.DataFrame({'% Missing':df[d].isnull().sum()/df[d].shape[0]*100})

        center = pd.DataFrame({'Mean':df[d].mean()}, index=d)        
        center['Mode'] = mode(df[d])[0].flatten()

        q5 = pd.DataFrame({'q5':df[d].apply(lambda x:x.dropna().quantile(0.05))})
        q10 = pd.DataFrame({'q10':df[d].apply(lambda x:x.dropna().quantile(0.10))})
        q25 = pd.DataFrame({'q25':df[d].apply(lambda x:x.dropna().quantile(0.25))})
        q50 = pd.DataFrame({'q50':df[d].apply(lambda x:x.dropna().quantile(0.50))})
        q75 = pd.DataFrame({'q75':df[d].apply(lambda x:x.dropna().quantile(0.75))})
        q95 = pd.DataFrame({'q95':df[d].apply(lambda x:x.dropna().quantile(0.95))})
        q99 = pd.DataFrame({'q99':df[d].apply(lambda x:x.dropna().quantile(0.99))})

        sd = pd.DataFrame({'SD':df[d].std()})
        se = pd.DataFrame({'SE':df[d].sem()})

        skew = pd.DataFrame({'Skew':df[d].skew(axis=0)})
        skew['Skewed'] = self._skew(skew)
        
        kurt = pd.DataFrame({'Kurtosis':df[d].kurtosis(axis=0)})
        kurt['Kurtosic'] = self._kurtosis(kurt)

        
        normal = pd.DataFrame({'Normality p-Value':df[d].apply(lambda x:shapiro(x.dropna())[1])})
        normal['Normality H_0'] = np.where(normal < sig, "Reject", "Do Not Reject")

        dq = pd.concat([obs,unique,missing,pct_missing,center,q5,q10,q25,
                        q50,q75,q95,q99,sd,se,skew,kurt,normal],
                        axis=1)
        return(dq)

    def describe(self, quantiles=None, include='all', exclude=None, sig=0.05):
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

            sig (float): The level of significance for the normality test.

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
                                                              exclude=exclude,
                                                              sig = sig)
            else:
                result['qualitative'] = self._describe_qual(self.x.to_frame(),
                                                            include=include,
                                                            exclude=exclude)

        else:
            # Describe qualitative columns
            qual = self.x.select_dtypes([np.bool, np.object])
            if qual.empty:
                pass
            else:
                result['qualitative'] = self._describe_qual(df=qual, include=include,
                                                    exclude=exclude)

            # Describe quantitative columns
            quant = self.x.select_dtypes([np.number])
            if quant.empty:
                pass
            else:
                result['quantitative'] = self._describe_quant(df=quant, 
                                                quantiles=quantiles, 
                                                include=include,
                                                exclude=exclude, sig=sig)
        self.description = result

        return(result)

    def factor_describe(self, df, x, y, z=None):
        """ Describes a quantitative variable y, by factors of x, and optional z

        Splits a dataframe along factors of the categorical variable, x and by
        an optional categorical variable z,  and returns descriptive statistics 
        of y, by factor.

        Arguments:
        ----------
            df (pd.DataFrame): Dataframe containing data
            x (str): The name of the categorical variable
            y (str): The name of the numeric variable
            z (str): The name of an optional categorical variable

        Returns:
        --------
        DataFrame: Descriptive statistics of y, by x and optionally by z.

        """
        df2 = pd.DataFrame()
        if z:
            gb = df[[x, y, z]].groupby([x, z])
            groups = [gb.get_group(x) for x in gb.groups]
            for g in groups:
                g_y = pd.DataFrame({y:g[y]})
                d = self._describe_quant(g_y)
                d[x] = g[x].unique()
                d[z] = g[z].unique()
                df2 = df2.append(d)
        else:
            gb = df[[x,y]].groupby([x])
            groups = [gb.get_group(x) for x in gb.groups]
            for g in groups:
                g_y = pd.DataFrame({y: g[y]})
                d = self._describe_quant(g_y)
                d[x] = g[x].unique()
                df2 = df2.append(d)
        return(df2)





