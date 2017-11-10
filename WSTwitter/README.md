# Study of tweets during World Series

This is a study of tweets containing the hashtag #WorldSeries that were collected during Game 6 of the 2017 World Series.  The tweets were collected from the Twitter streaming API (using tweepy).  Linguistic studies are performed, including word frequencies and sentiment analysis.  The sentiment analysis is done by training a Naive Bayesian classifier using labeled tweets from an NLTK corpus.
The geographical distribution of tweets is also studied.  The rate of tweets as a function of time is also investigated.

## Files

**CountyStudy.ipynb**:  A Jupyter notebook including python code, explanation, and results.  
The code has been tested in Python 3.  The data used for the analysis is not included in the
git folder.  The data was collected using the 

**twitterStreamToFile.py**: A python script which uses the tweepy package to collect
tweets using the Twitter streaming API. The tweets are saved to a text file in JSON format.

## Tools used
- Tweet collection: tweepy
- Data processing and analysis: Python, pandas
- Language analysis: nltk
- Machine learning/Sentiment analysis: nltk
- Visualization: matplotlib
- Geographical plots: Basemap