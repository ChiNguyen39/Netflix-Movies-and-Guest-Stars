#!/usr/bin/env python
# coding: utf-8

# ## 1. Loading data into a dictionary
# <p><img src="https://assets.datacamp.com/production/project_1237/img/netflix.jpg" alt="Someone's feet on table facing a television"></p>
# <p>Netflix! What started in 1997 as a DVD rental service has since exploded into the largest entertainment/media company by <a href="https://www.marketwatch.com/story/netflix-shares-close-up-8-for-yet-another-record-high-2020-07-10">market capitalization</a>, boasting over 200 million subscribers as of <a href="https://www.cbsnews.com/news/netflix-tops-200-million-subscribers-but-faces-growing-challenge-from-disney-plus/">January 2021</a>.</p>
# <p>This project is to analyse on a sample of Netflix data to answer the question: "Are movies getting shorter?" </p>
# <p>Firstly, Let's have a quick look at the following information. For the years from 2011 to 2020, the average movie durations are 103, 101, 99, 100, 100, 95, 95, 96, 93, and 90, respectively.</p>

# In[1]:
# Import pandas under its usual alias
import pandas as pd
import matplotlib.pyplot as plt
# Create the years and durations lists
years = [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
durations = [103, 101, 99, 100, 100, 95, 95, 96, 93, 90]

# Create a dictionary with the two lists
movie_dict = {'years': years, 'durations': durations}

# Print the dictionary
movie_dict


# ## 2. Creating a DataFrame from a dictionary
# <p>To convert our dictionary <code>movie_dict</code> to a <code>pandas</code> DataFrame, we will first need to import the library under its usual alias. We'll also want to inspect our DataFrame to ensure it was created correctly. Let's perform these steps now.</p>

# In[2]:
# Create a DataFrame from the dictionary
durations_df = pd.DataFrame(movie_dict)

# Print the DataFrame
durations_df


# ## 3. A visual inspection of our data
# <p>Alright, we now have a <code>pandas</code> DataFrame, now we want to start with a visualization of the data.</p>
# <p>Given that the data is continuous, a line plot would be a good choice, with the dates represented along the x-axis and the average length in minutes along the y-axis. This will allow us to easily spot any trends in movie durations.</p>

# In[3]:


# create a figure
fig = plt.figure()

# Draw a line plot of release_years and durations
plt.plot(years, durations)

# Create a title
plt.title('Netflix Movie Durations 2011-202')

# Show the plot
plt.show()


# ## 4. Loading the rest of the data from a CSV
# <p>Well, it looks like there is something to the idea that movie lengths have decreased over the past ten years! However with this data set, we're limited in the further explorations we can perform. There are a few questions about this trend that we are currently unable to answer, including:</p>
# <ol>
# <li>What does this trend look like over a longer period of time?</li>
# <li>Is this explainable by something like the genre of entertainment?</li>
# </ol>
# <p>Let's create another DataFrame, this time with all of the data.</p>

# In[4]:


# Read in the CSV as a DataFrame
netflix_df = pd.read_csv('/Users/kienguyen/Documents/Regis/Datacamp/01. DS course/02. Machine Learning with tree-based models/Project/datasets/netflix_data.csv')

# Print the first five rows of the DataFrame
netflix_df.head(5)


# ## 5. Filtering for movies!
# <p>Okay, we have our data! Now we can dive in and start looking at movie lengths. </p>
# <p>Or can we? Looking at the first five rows of our new DataFrame, we notice a column <code>type</code>. Scanning the column, it's clear there are also TV shows in the dataset! Moreover, the <code>duration</code> column we planned to use seems to represent different values depending on whether the row is a movie or a show (perhaps the number of minutes versus the number of seasons)?</p>
# <p>That's why we now select rows where <code>type</code> is <code>Movie</code>. While we're at it, we don't need information from all of the columns, so let's create a new DataFrame <code>netflix_movies</code> containing only <code>title</code>, <code>country</code>, <code>genre</code>, <code>release_year</code>, and <code>duration</code>.</p>
# 

# In[5]:


# Subset the DataFrame for type "Movie"
netflix_df_movies_only = netflix_df[netflix_df['type'] == 'Movie']

# Select only the columns of interest
netflix_movies_col_subset = netflix_df_movies_only[['title', 'country', 'genre', 'release_year', 'duration']]

# Print the first five rows of the new DataFrame
netflix_movies_col_subset.head(5)


# ## 6. Creating a scatter plot
# <p>This time, we are no longer working with aggregates but instead with individual movies. A line plot is no longer a good choice for the data, so I will try a scatter plot instead. We will again plot the year of release on the x-axis and the movie duration on the y-axis.</p>

# In[6]:


# Create a figure and increase the figure size
fig = plt.figure(figsize=(12, 8))

# Create a scatter plot of duration versus year
plt.scatter(netflix_movies_col_subset['release_year'], netflix_movies_col_subset['duration'])

# Create a title
plt.title('Movie Duration by Year of Release')

# Show the plot
plt.show()


# ## 7. Digging deeper
# <p>This is already much more informative than the simple plot I created before. We can also see that, while newer movies are overrepresented on the platform, many short movies have been released in the past two decades.</p>
# <p>Upon further inspection, something else is going on. Some of these films are under an hour long! Let's filter our DataFrame for movies with a <code>duration</code> under 60 minutes and look at the genres. This might give us some insight into what is dragging down the average.</p>

# In[7]:


# Filter for durations shorter than 60 minutes
short_movies = netflix_movies_col_subset[netflix_movies_col_subset['duration'] < 60]

# Print the first 20 rows of short_movies
short_movies.head(20)


# ## 8. Marking non-feature films
# <p>Interesting! It looks as though many of the films that are under 60 minutes fall into genres such as "Children", "Stand-Up", and "Documentaries". This is a logical result, as these types of films are probably often shorter than 90 minute Hollywood blockbuster. </p>
# <p>Now, let's explore the effect of these genres on the data by plotting them, and mark them with a different color.</p>

# In[8]:


# Define an empty list
colors = []

# Iterate over rows of netflix_movies_col_subset
for lab, row in netflix_movies_col_subset.iterrows():
    if row['genre'] == 'Children':
        colors.append('red')
    elif row['genre'] == 'Documentaries':
        colors.append('blue')
    elif row['genre'] == 'Stand-Up':
        colors.append('green')
    else:
        colors.append('black')
        
# Inspect the first 10 values in your list        
colors[:10]


# ## 9. Plotting with color!
# <p>We now have a <code>colors</code> list that we can pass to our scatter plot, which should allow us to visually inspect whether these genres might be responsible for the decline in the average duration of movies.</p>

# In[9]:


# Set the figure style and initalize a new figure
plt.style.use('fivethirtyeight')
fig = plt.figure(figsize=(12, 8))

# Create a scatter plot of duration versus release_year
plt.scatter(x=netflix_movies_col_subset['release_year'], y=netflix_movies_col_subset['duration'], c=colors)

# Create a title and axis labels
plt.title('Movie duration by year of release')
plt.xlabel('Release year')
plt.ylabel('Duration (min)')

# Show the plot
plt.show()


# ## 10. Conclusion
# <p>Well, as we suspected, non-typical genres such as children's movies and documentaries are all clustered around the bottom half of the plot. But we can't know for certain until we perform additional analyses. </p>
# <p>And after we performed an exploratory analysis of the data, we are certain that movies are getting shorter.</p>

# In[ ]:
