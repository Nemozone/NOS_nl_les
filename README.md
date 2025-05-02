# NOS Journal Dutch Learning App

This project is a Streamlit application specifically integrated with the "NOS Journal in makkelijke taal" YouTube channel. The app automatically pulls the top 5 videos from the channel, allowing users to choose one. It then uses the video's transcript to create fully educational content to help users learn new Dutch vocabulary and grammar. Additionally, it generates extra exercises for better understanding and practice.

## Project Structure

```
streamlit-youtube-app
├── src
│   ├── tools.py          # Contains functions for retrieving transcript text and agents to create lessons
│   └── app.py            # Main entry point for the Streamlit application
├── requirements.txt      # Lists the dependencies required for the project
└── README.md             # Documentation for the project
```

## Setup Instructions

1. Clone the repository:
```
<repository-url>
cd streamlit-youtube-app
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit application:
```
streamlit run src/app.py
```

2. Open your web browser and go to `http://localhost:8501`.

3. The app will automatically pull the top 5 videos from the
the NOS channel and display them in the sidebar. Once a video is selected, the app will show the video and automatically generate a series of learning materials, including:

• Translation of key phrases  
• Vocabulary lists with definitions  
• Explanations of key grammar points  
• Additional exercises for practice  