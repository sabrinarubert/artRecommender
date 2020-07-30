# Import Supporting Libraries
import pandas as pd
import numpy as np
import math

# Import Dash Visualization Libraries
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, MATCH, ALL
import base64

# import SQL Libraries
from sqlalchemy import create_engine
import pymysql

# import ML Libraries
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

app = dash.Dash(__name__)

# -----------------------------------
# setup layout vars
state = "choose"
layout_choose = None
layout_display = None

# -----------------------------------
#setup SQL connection
sqlEngine       = create_engine('mysql+pymysql://root:sabrina_art@127.0.0.1:3301/sabrina_art')
dbConnection    = sqlEngine.connect()

# -----------------------------------
# load features
features_source = pd.read_sql("""

SELECT DISTINCT artwork_id, gene_id, num_genes_of_artwork  FROM

(SELECT artwork_id as 'id', COUNT(DISTINCT gene_id) AS 'num_genes_of_artwork'
FROM artworks_to_genes
GROUP BY artwork_id) T

LEFT JOIN artworks_to_genes ON T.id = artworks_to_genes.artwork_id

WHERE num_genes_of_artwork > 3
ORDER BY num_genes_of_artwork DESC""", dbConnection)

features_source["rating"] = 1
artFeatures = features_source.pivot_table(columns="gene_id",index="artwork_id",values="rating").fillna(0)

def loadRandomImages():
    global dbConnection
    randomImages = pd.read_sql("""SELECT
    DISTINCT artwork_id,
	image
FROM
	(
		SELECT
			DISTINCT artwork_id,
			gene_id,
			num_genes_of_artwork
		FROM
			(
				SELECT
					artwork_id AS 'id',
					COUNT(DISTINCT gene_id) AS 'num_genes_of_artwork'
				FROM
					artworks_to_genes
				GROUP BY
					artwork_id
			) T
			LEFT JOIN artworks_to_genes ON T.id = artworks_to_genes.artwork_id
		WHERE
			num_genes_of_artwork > 3
		ORDER BY
			num_genes_of_artwork DESC
	) N
	LEFT JOIN artworks ON N.artwork_id = artworks.id
ORDER BY
	RAND ()
LIMIT
	12""", dbConnection)
    return randomImages

# -----------------------------------
# fit model
mat_artFeatures = csr_matrix(artFeatures)

model_knn = NearestNeighbors(metric="cosine",
                             algorithm="brute",
                             n_jobs=-1)
model_knn.fit(mat_artFeatures)

# -----------------------------------
# functions for loading nearest neighbors

def getListOfRemmonendations(artwork):
    global artFeatures
    result = pd.DataFrame(columns=['indice', 'distance'])
    
    #distances, indices = model_knn.kneighbors(artwork, n_neighbors=len(artFeatures))
    distances, indices = model_knn.kneighbors(artwork, n_neighbors=10)

    for i in range(1, len(distances[0])):
        distance =  distances[0][i]   
        indice =    indices[0][i]
        result = result.append({'distance': distance, 'indice': indice},ignore_index=True)
    
    return result
    
def combineArtworks(artworks):
    
    result = pd.DataFrame(columns=['indice', 'distance'])
    
    for artwork_id in artworks:
        artwork_features = artFeatures.loc[artwork_id].values.reshape(1, -1)
        result = result.append(getListOfRemmonendations(artwork_features),ignore_index=True)
    
    return result

# -----------------------------------
# html components

back_button = html.Div(
    className="button_row_box",
        children = [
            html.Button('try again', className="grey_button", id="back")
        ],
        id={
            'type': 'back_button',
            'index': 'buttons'
        }
    )

like_button = html.Div(
    className="button_row_box",
        children = [
            html.Button("find artworks i like", className="green_button", id="artwork_i_like")
        ],
        id={
            'type': 'like_button',
            'index': 'buttons'
        }
    )

dont_like_button = html.Div(
    className="button_row_box",
        children = [
            html.Button("find artworks i don't like", className="red_button", id="artwork_i_dont_like")
        ],
        id={
            'type': 'dont_like_button',
            'index': 'buttons'
        }
    )

header = html.Div(
                    id="header",
                    children=[
                        html.H2(children="find art you like"),
                        html.H3(
                            id="description"
                        ),
                    ],
                )

# -----------------------------------
# content functions

def buildContentChoose():

    content = []
    randomImages = loadRandomImages() 
    images = len(randomImages)

    if images % 3 > 0:  
        rows = math.ceil(images / 3)
    else:
        rows = int(images / 3)

    for i in range (0, rows):

        if (images % 3) > 0 and (i == rows -1):
            imagesInRow = images % 3
        else:
            imagesInRow = 3

        rowChildren = []

        for k in range (0, imagesInRow):
         
            indexOfImage = (i*3)+k
            imagePath = randomImages.iloc[indexOfImage]["image"]
            artwork_id = randomImages.iloc[indexOfImage]["artwork_id"]
            
            image = html.Div(
                        className="column",
                        children=[
                            html.Img(src = imagePath),
                            dcc.Checklist(
                                    className="checkbox",
                                    options=[{"label": " ", "value": artwork_id}],
                                    value=[],
                                    id={
                                        'type': 'checkbox',
                                        'index': "str(indexOfImage)"
                                    }
                                )
                    ])
            
            rowChildren.append(image)

        imageRow = html.Div(className="row",children=rowChildren)
        content.append(imageRow)

    return content

# -----------------------------------
# App layout display
def buildContentDisplay(neighbors):
    global dbConnection

    content = []
    
    # get artworks (neighbors)
    for index, neighbor in neighbors.iterrows():
    
        artworkIndex = int(neighbor.name)
        artworkId = artFeatures.iloc[artworkIndex].name
        
        artworkList = pd.read_sql("select * from artworks where id LIKE '"+artworkId+"' Limit 1", dbConnection)
        recommended_artwork = artworkList.iloc[0]
        
        # get all information about the artwork
        artwork_title = recommended_artwork["title"]
        artwork_id = recommended_artwork["id"]
        artwork_image = recommended_artwork["image"]
        artwork_date = recommended_artwork["date"]
        artwork_medium = recommended_artwork["medium"]
        artwork_artists = recommended_artwork["artists"]

        details = [html.P(artwork_title),
                    html.P(artwork_medium),
                    html.P(artwork_date)]


        # get artist name
        artistList = pd.read_sql("""SELECT * 
        FROM artists
        LEFT JOIN artworks_to_artist ON artworks_to_artist.artists_id = artists.id
        WHERE artworks_to_artist.artwork_id LIKE '"""+artworkId+"""' Limit 1""", dbConnection)


        if (len(artistList) > 0):
        
            artist_from_artwork = artistList.iloc[0]
            artist_name = artist_from_artwork["name"]
            artist_birthday = artist_from_artwork["birthday"]
            artist_deathday = artist_from_artwork["deathday"]
            artist_hometown = artist_from_artwork["hometown"]

            if (artist_name != None and artist_birthday != None and artist_deathday != None and artist_hometown != None):
                details.append(html.P("by " +  artist_name + " (from: "+artist_birthday +" to "+ artist_deathday+")"))
                details.append(html.P(artist_hometown))

        # build html for rows
        imageRow = html.Div(
            className="row",
            children=[
                html.Div(
                    className="column",
                    children=[html.Img(src =artwork_image)
                ]),
                html.Div(
                    className="columnResultDescription",
                    children=details)
            ])

        content.append(imageRow)

    return content

# -----------------------------------
# start application
app.layout = html.Div(id="root",children=[header, html.Div(id="content"), html.Br(),  html.Div(id="buttons")])

# -----------------------------------
# call backs
@app.callback(
            [
                Output('description', 'children'),
                Output('content', 'children'),
                Output('buttons', 'children')
            ],
            [
                Input({'type': 'back_button', 'index': ALL}, 'n_clicks'),
                Input({'type': 'like_button', 'index': ALL}, 'n_clicks'),
                Input({'type': 'dont_like_button', 'index': ALL}, 'n_clicks'),
            ],
            [State({'type': 'checkbox', 'index': ALL}, 'value')])
def update_output(back_button_clicked, like_button_clicked, dont_like_button_clicked, listOfValues):
    global layout_choose
    global layout_display
    global state

    # collect list of favorite Artworks
    favoriteArtworks = []
    for checkBox in listOfValues:
        if (len(checkBox) >0):
            favoriteArtworks.append(checkBox[0])

    content = []

    if (len(like_button_clicked) > 0 and like_button_clicked[0] == 1):
        
        combinedRecommendations = combineArtworks(favoriteArtworks)
        combinedRecommendations["distance"] = 1- combinedRecommendations["distance"]
        combinedRecommendations.reindex()
        neighbors = combinedRecommendations.groupby(['indice']).sum().sort_values("distance", ascending = False).head(10)

        content =  buildContentDisplay(neighbors)
        state = "display"

    elif (len(dont_like_button_clicked) > 0 and dont_like_button_clicked[0] == 1):

        combinedRecommendations = combineArtworks(favoriteArtworks)
        combinedRecommendations["distance"] = 1- combinedRecommendations["distance"]
        combinedRecommendations.reindex()
        neighbors = combinedRecommendations.groupby(['indice']).sum().sort_values("distance", ascending = True).head(10)

        content =  buildContentDisplay(neighbors)
        state = "display"
    
    elif (len(back_button_clicked) > 0 and back_button_clicked[0] == 1):
        content =  buildContentChoose()
        state = "choose"
    else:
        content =  buildContentChoose()
        state = "choose"

    buttons = []

    if (state == "choose"):
        description="please choose a few pictures that you like. the recommender will then suggest artworks similiar to those that you have chosen."
        buttons = [like_button]

    elif (state == "display"):
        description="i am sure that you like those pictures:"
        buttons = [back_button]

    return description, content, buttons

if __name__ == '__main__':
    app.run_server(debug=False)