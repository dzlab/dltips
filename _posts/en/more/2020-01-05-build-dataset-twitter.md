---
layout: post

title: Build a dataset from tweets

tip-number: 03
tip-username: dzlab
tip-username-profile: https://github.com/dzlab
tip-tldr: Some new social movement have emerged on social media, how could get enough data to study/undestand what's happening?
tip-writer-support: https://www.patreon.com/dzlab

categories:
    - en
    - more
---

Twitter is a huge source of valuable data as people from around the world use Twitter to post billion of tweets daily about anything and everything. Collecting and analysing such data gives a tremendous amount of insights that have completely transformed many reaserch fields. For instance, social studies used to use historical text written by intellecuals and never as now were able to analyze social movements in real time and directly from the people involved. Also, businesses now can leaverage it to gather feedback about their company, brand, product, service, etc.

In Python, we can use the `tweepy` library which can be installed with `pip install tweepy`.
Before to be able to query data, we need to create a Twitter developer account then obtain user tokens ([follow these steps](https://developer.twitter.com/en/docs/basics/authentication/oauth-1-0a/obtaining-user-access-tokens)).

* `CONSUMER_KEY`: consumer key is associated with the application (Twitter, Facebook, etc.).
* `CONSUMER_SECRET`: consumer secret is used to authenticate with the authentication server (Twitter, Facebook, etc.).
* `ACCESS_TOKEN`: access token is given after successful authentication of above keys
* `ACCESS_TOKEN_SECRET`: access token secret is a sort of the password associated with the access key.

Now we can use those keys to authoticate our selves to the Twitter API

```python
import tweepy

auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
```

The Twitter API can be either directly queried to get a one time response

```python
api = tweepy.API(auth)
Tweets = api.search('some search query', count = 10, lang='en', exclude='retweets', tweet_mode='extended')
```

The API also provides a way to register a listener that will be notified everytime the search criteria matched, e.g. hashtag match.

First define the body of the listener implementation, e.g. open a file and append tweets text
```python
import json
import os

class StdOutListener(tweepy.streaming.StreamListener):
  def __init__(self, filename, total=1e4):
    self.out_file = open(filename,'a')
    self.out_file.write("tweet_text" + os.linesep)
    super().__init__()

  def on_data(self, data):
    """Handle a new tweet received on the stream"""
    tweet_obj = json.loads(data)
    tweet = tweet_obj["text"]
    username = tweet_obj["user"]["screen_name"]
    try:
      if 'extended_tweet' in j.keys():
        text = tweet_obj['extended_tweet']['full_text']
        self.out_file.write('"' + text + '"' + os.linesep)
    except KeyError:
      pass

  def on_error(self, status):
    print("ERROR")
    print(status)
```

Then using Twitter Stream API as follows:

```python
try:
  tweets_listener = StdOutListener('tweets.csv')
  stream = tweepy.Stream(auth, tweets_listener)
  stream.filter(languages=["en"], track=['en'])
except KeyboardInterrupt:
  pass
```