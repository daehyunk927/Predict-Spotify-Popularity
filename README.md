# Spotify

## Will My Song Be Spotify Famous?

What makes a song popular is a complex question. Undoubtedly, external social factors
such as the level of fame of an artist or a song’s social significance can impact its
popularity. Internal musical attributes of a song, however, also play a critical role to its
commercial success.
Having the ability to pre-determine the popularity of a song based on its musical
attributes could be useful to singers, songwriters, and labels, and could help them
decide the direction of their music.
How do we break a song down into its “musical attributes”? Currently, Spotify parsing
software is able to break down a song into 13 musical attributes.
Our dataset contains 19000 songs that have gone through this parsing process.
Our goal was to see how musical features themselves can predict a song’s popularity.
We did this by using the 13 audio features as our predictor variables. We used Spotify’s
“Song Popularity” variable as our response variable. Song popularity is a numeric value
between 0-100 determined based on the number of plays and recentness of plays a
song has received. For ease of use, the popularity in our model has been binned into 5
categories, 1-5, with being the most popular. Several predictive models of various types
(Linear SVC, Random Forest, K Nearest Neighbors, etc.) were tested and the most
accurate one was selected.
Lastly, we built a Flask App with HTML user interface such that a user can manually
input the audio features of their own original song (i.e. a song that does not exist in our
19000 song data set) and predict its popularity. The UI has two components. First, a
user can look up familiar songs using a search feature that connects to a Spotify API;
the API will then return the 13 musical attributes of the known song. This allows the user
to become familiar with the attributes. Using this information, the user can then input
their estimates for an unknown or original song for each of the 13 attributes. These
values are then run through our model and the predicted popularity of the song is
returned to the user.
