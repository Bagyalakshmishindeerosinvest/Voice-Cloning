# 🎙️ Voice Cloning Web App

This is a Flask-based Voice Cloning Web Application that allows users to input a short reference audio clip (uploaded or recorded) and custom text to generate speech in the same voice using AI models.

---

## 🔊 About Voice Cloning

Voice cloning is an advanced AI technique that mimics a person’s voice by analyzing a short speech sample. This technology can generate new audio outputs using the same voice for completely new phrases or sentences. It has applications in:

- 🎧 Personalized virtual assistants and characters  
- 🗣️ Giving voice to people with speech impairments  
- 🎬 Voiceovers for videos and animations  
- 🎓 Educational tools with character voices  

---

## ⚙️ Setup Instructions

Follow the steps below to set up and run the application on your local machine.

### 1. Clone the Repository

## 2. Create a Virtual Environment (Python 3.8)

Ensure Python 3.8 is installed. Then create and activate a virtual environment:

```bash
python3.8 -m venv venv
source venv/bin/activate
# On Windows: venv\Scripts\activate


## 3. Install Dependencies

Install required packages from the `requirements.txt` file:


pip install -r requirements.txt

## 4. Run the Application

Once everything is ready, run the app using:

python new.py

###  Download Pretrained Models
Pretrained models are now downloaded automatically. If this doesn't work for you,
 you can manually download them
 [here](https://github.com/CorentinJ/Real-Time-Voice-Cloning/wiki/Pretrained-models).

📁 **Make sure to place the downloaded weights in the following structure:**  
`saved_models → default → *.pth` (your model weight files should go inside the `default` folder).



