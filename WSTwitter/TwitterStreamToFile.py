# -*- coding: utf-8 -*-

import json
import tweepy
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy.streaming import StreamListener

import time


class MyTwitterListener(StreamListener):
    
    def on_status(self, status):
        print("Status = ",status.text)

    def on_error(self, status):
        print("Got an error: ", status)
        return True
    
    def on_data(self, data):
        try:

            f = open('worldSeriesG7-5.json','a')  #['#WorldSeries']
            f.write(data)
            
            #datadict = json.loads(data)
            
        except BaseException as myError:
            print("Error in on_data: ",str(myError))
            time.sleep(5)
        return True


#Private information kept secret.  Users will need to provide their own Twitter keys/tokens.
consumerKey = 'SECRET'
consumerSecret = 'SECRET'
accessToken = 'SECRET'
accessSecret = 'SECRET'

auth = OAuthHandler(consumerKey, consumerSecret)
auth.set_access_token(accessToken, accessSecret)

myStream = tweepy.Stream(auth = auth, listener=MyTwitterListener())

print("Starting stream...")
myFilterList = ['#WorldSeries']


myStream.filter(track=myFilterList)


