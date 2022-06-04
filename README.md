# songnet: Song Recommendation Network

songnet is sequence model that gives personalized song recommendations based on the listening patterns
of a specific user.  

## Motivation

When listening to a particular song on Spotify, I find myself thinking about "related" songs. It is hard to quantitatively
descrive "related". It needs to be understood that "related" is not necessarliy "similar" (which is the way Spotify does it). For example, sometimes I go from one genre of songs
to another, other times I go from happy songs to sad songs by. Usually, I would just search for the song and listen. I wanted something
that can generate personalized playlist(s) given initial few songs based on my listening patterns. This project is an attempt to solve this problem of learning listening patterns of a user. 

## Limitations

Currently, this project uses two things from Spotify:

*   Song Features (Easy to get using Spotify API)
*   Spoitfy Streaming History (Not possible to get using Spotify API)

The only way to get your streaming history is by requesting Spotify your personal data and they process it and send
it back to you after a few days. Since it is currently not possible to extract this streaming history from the API,
making this network generic is a lot harder than usual for an end-user. Although, the code itself can be used to train
the model for any particular user given that they provide their streaming history. There is no active maintainence to make this code ready to use for any end-user.

## Getting Started

For attempting to train the network with your own streaming history, take a look at [train_spotify.py](train_spotify.py)

## Details

The streaming data is converted into variable length sequences of songs. To mitigate the problem of long-term streaming activity,
we only consider a song sequence to be a sequence if they were streamed in an single hour. This time-span choice is arbitrary but
can be tuned personally based on the user themselves. 

The model is a simple LSTM based sequence model, that takes in sequences of songs to generate a target song. Each song in the sequence
is represented by its features extracted from the Spotify API such as acousticness, valence etc. 

Once the model is trained, we generate personalized playlist by bootstrapping the model by giving initial few songs. And then use
cosine similarity matrix to add more songs to the sequence. 

## Future Work

Spotify API features while being easy to get are arbitrary i.e. they are based on expert knowledge and are hand-crafted. To achieve the
true power of Deep-Learning, we must fingerprint i.e. extract features of a song based on its waveform. There are some scripts
that try to this but are currently very limited and do not perform well. 

Also, if possible, the language of the song should be added as
a feature because in my personal experience, streaming multi-language songs in a single session is rare. So the model needs to make an informed decision while generating recommendation