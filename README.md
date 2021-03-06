# Art Recommender

*Sabrina Rubert, July 2020, Berlin*

## Content
* [Project Description](https://github.com/sabrinarubert/artRecommender#project-description)
* [Question](https://github.com/sabrinarubert/artRecommender#question)
* [Dataset](https://github.com/sabrinarubert/artRecommender#dataset)
* [Workflow](https://github.com/sabrinarubert/artRecommender#workflow)
* [Recommender System](https://github.com/sabrinarubert/artRecommender#recommender-system)
* [Conclusion](https://github.com/sabrinarubert/artRecommender#conclusion)
* [Limitations](https://github.com/sabrinarubert/artRecommender#limitations)
* [Links](https://github.com/sabrinarubert/artRecommender#links)

## Project Description
As nowadays, social media and other online tools and networks have given people platforms to articulate their preferences, people comment, like and share a lot. Markets have responded by modulating their offering to our tastes. With for example Netflix, Amazon, Spotify and endless curated content blogs and websites we now inhabit a world where limitless tailored content is available. But what we have overlooked is an unchanging part of the cultural life: the art world hasn't changed.

That is why I decided to concentrate on that kinf of neglected market and try to find a why to makr art more accessible to not only a wider audience but also to make it more appealing to younger people.

## Question
How to make art more accessible to a wider audience, and especially more appealing to younger people?

## Datset
I got all relevant data from the [Artsy API](https://developers.artsy.net).
For the art recommender I needed data about the artists, the artworks and the genes.

Genes: Artsy innitiated the Art Genome Project, a classification system and technological framework that powers Artsy. It maps characterstics (Artsy calls them "genes") that connect artists, artowroks, architecture, and design objects across history. There are currently over 1,000 characteristics in the Art Genome Project, including art historical movements, subject matter, and formal qualities.

My final dataset consists of 1,033 genes and 30,000 artworks from the 7th century until the year 1917.

## Worflow
The following workflow was implemented:

1. Deciding on a topic for the project
2. Choosing and finding relevant data
3. Getting and cleaning the data
4. Building the model
5. Devloping a prototype
6. Creating the presentation

## Recommender System
Using the described dataset about artists, artworks and genes I build a recommender system using the k-nearest neighbors algorithm. The basis of the recommender system are the different genes that are assigned to various artworks.

The user chooses the artworks he/she likes most. The recommender then suggests artworks similar to the ones the user likes based on their characterstics.

## Conclusion
The art recommender prototype:
* Develops an understanding about what qualtities of visual art users like and dislike
* Gives users a digital space to view art that they have not specifically searched for
* Provides personalized content while the users are having an interactive art experience

## Limitations
Due to public rights it is not possible to access all artworks from the Artsy API. Therefore only artworks from the 7th century until thee year 1916 were used within this art recommender. Looking forward the art recommender could be extended adding more current artwork.

## Links
This repository is divided into four different folders.

**00_Data**
* All data files and data information are within this [folder](https://github.com/sabrinarubert/artRecommender/tree/master/00_Data)

**01_Notebooks**
* All notebooks for getting the various data from the Artsy API are found [here](https://github.com/sabrinarubert/artRecommender/tree/master/01_Notebooks)
* The final notebook for the art recommender: [art_recommender.ipynb](https://github.com/sabrinarubert/artRecommender/blob/master/01_Notebooks/art_recommender.ipynb)

**02_App**
* The code for the art recommender prototype as well as all assets can be found [here](https://github.com/sabrinarubert/artRecommender/tree/master/02_App)

**03_Presentation**
* Google Slides: [200731_ArtRecommender](https://github.com/sabrinarubert/artRecommender/blob/master/03_Presentation/200731_ArtRecommender.pdf)
