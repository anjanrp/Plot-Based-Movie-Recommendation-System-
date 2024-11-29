# Plot-Based-Movie-Recommendation-System-
This was done as a part of the coursework Topics in Deep Learning in PES University in May 2024

This Streamlit application predicts the genre of movies based on their plot summaries obtained from Wikipedia and recommends similar movies. It utilizes a machine learning model trained on movie plot data to perform genre classification and provides movie recommendations accordingly.

Features

Genre Prediction: Predict the genre of a movie by its plot summary.
Movie Recommendations: Recommend similar movies based on the predicted genre.
Custom Styles: Enhanced user interface with custom styles and background.


Input: The user is prompted to enter the name of a movie.
Processing:
    The application queries the Wikipedia page for the entered movie name to fetch the plot summary.
    It processes the plot summary through a pre-trained neural network to predict the genre.
    Based on the predicted genre, the application queries an internally maintained dataset to recommend similar movies.

Steps to execute :
Put all the files here in the same directory -> Run the model training file -> Launch the Streamlit application by the command - streamlit run run.py. 

Note : 
    The model training code is optimised for running on the M Generation Macbooks;
    Streamlit should be setup on the system where the application is being run. 
